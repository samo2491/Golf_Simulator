package com.golfsim.app.camera

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.PointF
import android.hardware.camera2.CaptureRequest
import android.util.Range
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.core.*
import android.hardware.camera2.CameraMetadata
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.golfsim.app.models.BallPosition
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sign
import kotlin.math.sqrt
import kotlin.math.tan

class GolfCameraManager(private val context: Context) {

    companion object {
        private const val TARGET_FPS = 120
        private const val FRAME_HISTORY_SIZE = 160
        private const val VELOCITY_WINDOW = 10
        private const val MAX_GAP_FRAMES = 5
        private const val CAMERA_HFOV_DEG = 77.0

        private const val DOWNSAMPLE = 2
        private const val BG_ALPHA = 0.06f
        private const val MOTION_MIN_DELTA = 14f
        private const val LUMA_PERCENTILE = 0.84f
    }

    private val _trackingState = MutableStateFlow<TrackingState>(TrackingState.Idle)
    val trackingState: StateFlow<TrackingState> = _trackingState

    private val _detectedBallPositions = MutableStateFlow<List<BallPosition>>(emptyList())
    val detectedBallPositions: StateFlow<List<BallPosition>> = _detectedBallPositions

    private val _swingDetected = MutableStateFlow(false)
    val swingDetected: StateFlow<Boolean> = _swingDetected

    private val _fps = MutableStateFlow(0f)
    val fps: StateFlow<Float> = _fps

    private val _ballRadius = MutableStateFlow(0f)
    val ballRadius: StateFlow<Float> = _ballRadius

    private val _shotSnapshot = MutableStateFlow<ShotDataSnapshot?>(null)
    val shotSnapshot: StateFlow<ShotDataSnapshot?> = _shotSnapshot

    sealed class TrackingState {
        object Idle : TrackingState()
        object WaitingForBall : TrackingState()
        data class BallDetected(val x: Float, val y: Float, val radius: Float, val confidence: Float) : TrackingState()
        object SwingInProgress : TrackingState()
        object AnalyzingShot : TrackingState()
        data class Error(val message: String) : TrackingState()
    }

    @Volatile private var profile = CalibrationProfile()

    private var cameraProvider: ProcessCameraProvider? = null
    private val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    private val ballHistory = mutableListOf<BallPosition>()
    private var isCapturing = false
    private var motionFrameCount = 0
    private var gapFrameCount = 0

    @Volatile private var lastYData: ByteArray? = null
    @Volatile private var lastFrameWidth = 1920
    @Volatile private var lastFrameHeight = 1080

    private val frameTimestamps = ArrayDeque<Long>(128)

    private var downWidth = 0
    private var downHeight = 0
    private var backgroundLuma = FloatArray(0)
    private var currentLuma = FloatArray(0)
    private var motionMask = BooleanArray(0)
    private var brightMask = BooleanArray(0)

    private var kalmanX = 0f
    private var kalmanY = 0f
    private var kalmanVx = 0f
    private var kalmanVy = 0f
    private var kalmanInitialized = false
    private var kalmanP = Array(4) { i -> FloatArray(4) { j -> if (i == j) 100f else 0f } }
    private val processNoise = 4f
    private val measureNoise = 8f

    fun updateCalibration(newProfile: CalibrationProfile) {
        profile = newProfile
    }

    fun setManualBallHint(normX: Float, normY: Float) {
        profile = profile.copy(manualHintX = normX, manualHintY = normY, useManualHint = true)
    }

    suspend fun autoDetectBrightness(): Int {
        val frame = lastYData ?: return 200
        val width = lastFrameWidth
        val height = lastFrameHeight
        var peak = 0

        for (y in height / 5 until (height * 4) / 5 step 3) {
            for (x in width / 5 until (width * 4) / 5 step 3) {
                val index = y * width + x
                if (index < frame.size) {
                    peak = max(peak, frame[index].toInt() and 0xFF)
                }
            }
        }
        return (peak * 0.78f).toInt().coerceIn(130, 245)
    }

    @SuppressLint("UnsafeOptInUsageError")
    fun startCamera(
        previewView: PreviewView,
        lifecycleOwner: LifecycleOwner,
        onBallDetected: ((BallPosition) -> Unit)? = null
    ) {
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            cameraProvider = future.get()

            val previewBuilder = Preview.Builder().setTargetAspectRatio(AspectRatio.RATIO_16_9)
            Camera2Interop.Extender(previewBuilder).apply {
                setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(TARGET_FPS, TARGET_FPS))
                setCaptureRequestOption(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
                setCaptureRequestOption(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
                setCaptureRequestOption(CaptureRequest.CONTROL_AWB_LOCK, true)
                setCaptureRequestOption(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF)
                setCaptureRequestOption(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE, CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF)
                setCaptureRequestOption(CaptureRequest.SENSOR_EXPOSURE_TIME, 1_000_000L)
            }

            val preview = previewBuilder.build().also { it.setSurfaceProvider(previewView.surfaceProvider) }

            val analysisBuilder = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
            Camera2Interop.Extender(analysisBuilder)
                .setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(TARGET_FPS, TARGET_FPS))

            val analyzer = analysisBuilder.build().also {
                it.setAnalyzer(cameraExecutor) { frame -> processFrame(frame, onBallDetected) }
            }

            try {
                cameraProvider?.unbindAll()
                cameraProvider?.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    analyzer
                )
                _trackingState.value = TrackingState.WaitingForBall
            } catch (e: Exception) {
                _trackingState.value = TrackingState.Error("Camera failed: ${e.message}")
            }
        }, ContextCompat.getMainExecutor(context))
    }

    fun startCapturing() {
        synchronized(ballHistory) { ballHistory.clear() }
        motionFrameCount = 0
        gapFrameCount = 0
        kalmanInitialized = false
        _swingDetected.value = false
        _shotSnapshot.value = null
        isCapturing = true
        _trackingState.value = TrackingState.WaitingForBall
    }

    fun stopCapturing(): List<BallPosition> {
        isCapturing = false
        _trackingState.value = TrackingState.AnalyzingShot
        return synchronized(ballHistory) { ballHistory.toList() }
    }

    fun resetTracking() {
        synchronized(ballHistory) { ballHistory.clear() }
        motionFrameCount = 0
        gapFrameCount = 0
        kalmanInitialized = false
        _swingDetected.value = false
        _detectedBallPositions.value = emptyList()
        _shotSnapshot.value = null
        _trackingState.value = TrackingState.WaitingForBall
    }

    fun shutdown() {
        cameraExecutor.shutdown()
        cameraProvider?.unbindAll()
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun processFrame(imageProxy: ImageProxy, onBallDetected: ((BallPosition) -> Unit)?) {
        val now = System.currentTimeMillis()
        updateFps(now)

        val image = imageProxy.image ?: run {
            imageProxy.close()
            return
        }

        val width = imageProxy.width
        val height = imageProxy.height
        lastFrameWidth = width
        lastFrameHeight = height

        val plane = image.planes[0]
        val yData = ByteArray(plane.buffer.remaining()).also { plane.buffer.get(it) }
        lastYData = yData

        ensureBuffers(width, height)
        buildLumaMaps(yData, plane.rowStride, plane.pixelStride)

        val candidate = detectBallCandidate(width, height)
        if (candidate != null) {
            gapFrameCount = 0
            val filtered = kalmanUpdate(candidate.cx, candidate.cy)
            val pos = BallPosition(filtered.x, filtered.y, now)

            if (isCapturing) {
                synchronized(ballHistory) {
                    ballHistory.add(pos)
                    if (ballHistory.size > FRAME_HISTORY_SIZE) ballHistory.removeAt(0)
                }
                _detectedBallPositions.value = synchronized(ballHistory) { ballHistory.toList() }
                onBallDetected?.invoke(pos)
                checkForSwing()
            }

            _ballRadius.value = candidate.radius
            _trackingState.value = TrackingState.BallDetected(filtered.x, filtered.y, candidate.radius, candidate.confidence)
        } else {
            if (kalmanInitialized && gapFrameCount < MAX_GAP_FRAMES) {
                gapFrameCount++
                kalmanX += kalmanVx
                kalmanY += kalmanVy
                val predicted = BallPosition(kalmanX, kalmanY, now)
                if (isCapturing) synchronized(ballHistory) { ballHistory.add(predicted) }
                _trackingState.value = TrackingState.BallDetected(kalmanX, kalmanY, _ballRadius.value, 0.20f)
            } else {
                kalmanInitialized = false
                if (!isCapturing) _trackingState.value = TrackingState.WaitingForBall
            }
        }
        imageProxy.close()
    }

    private fun ensureBuffers(width: Int, height: Int) {
        val targetW = width / DOWNSAMPLE
        val targetH = height / DOWNSAMPLE
        if (targetW == downWidth && targetH == downHeight) return

        downWidth = targetW
        downHeight = targetH
        val size = downWidth * downHeight
        backgroundLuma = FloatArray(size)
        currentLuma = FloatArray(size)
        motionMask = BooleanArray(size)
        brightMask = BooleanArray(size)
        kalmanInitialized = false
    }

    private fun buildLumaMaps(yData: ByteArray, rowStride: Int, pixelStride: Int) {
        if (downWidth == 0 || downHeight == 0) return

        var offset = 0
        val histogram = IntArray(256)

        for (y in 0 until downHeight) {
            val srcY = y * DOWNSAMPLE
            for (x in 0 until downWidth) {
                val srcX = x * DOWNSAMPLE
                val rawIndex = srcY * rowStride + srcX * pixelStride
                val luma = if (rawIndex < yData.size) (yData[rawIndex].toInt() and 0xFF).toFloat() else 0f
                currentLuma[offset] = luma
                histogram[luma.toInt().coerceIn(0, 255)]++
                offset++
            }
        }

        val percentileLuma = percentile(histogram, (downWidth * downHeight * LUMA_PERCENTILE).toInt())
        val brightnessCutoff = max(profile.brightnessThreshold.toFloat(), percentileLuma)

        for (i in currentLuma.indices) {
            if (backgroundLuma[i] == 0f) backgroundLuma[i] = currentLuma[i]
            val delta = abs(currentLuma[i] - backgroundLuma[i])
            motionMask[i] = delta >= MOTION_MIN_DELTA
            brightMask[i] = currentLuma[i] >= brightnessCutoff
            backgroundLuma[i] = backgroundLuma[i] * (1f - BG_ALPHA) + currentLuma[i] * BG_ALPHA
        }
    }

    private fun percentile(hist: IntArray, targetCount: Int): Float {
        var cumulative = 0
        for (i in hist.indices) {
            cumulative += hist[i]
            if (cumulative >= targetCount) return i.toFloat()
        }
        return 255f
    }

    private data class Candidate(
        val cx: Float,
        val cy: Float,
        val radius: Float,
        val area: Float,
        val circularity: Float,
        val contrast: Float,
        val confidence: Float
    )

    private fun detectBallCandidate(frameWidth: Int, frameHeight: Int): Candidate? {
        val size = downWidth * downHeight
        if (size <= 0) return null

        val visited = BooleanArray(size)
        val candidates = ArrayList<Candidate>()

        val minAreaDown = (profile.minBallRadius * profile.minBallRadius * PI / (DOWNSAMPLE * DOWNSAMPLE)).toFloat() * 0.4f
        val maxAreaDown = (profile.maxBallRadius * profile.maxBallRadius * PI / (DOWNSAMPLE * DOWNSAMPLE)).toFloat() * 1.8f

        for (y in 1 until downHeight - 1) {
            for (x in 1 until downWidth - 1) {
                val idx = y * downWidth + x
                if (visited[idx] || !brightMask[idx]) continue

                val blob = collectComponent(x, y, visited, minAreaDown, maxAreaDown)
                if (blob != null) candidates.add(blob)
            }
        }

        if (candidates.isEmpty()) return null

        val hintedX = if (profile.useManualHint) profile.manualHintX * frameWidth else null
        val hintedY = if (profile.useManualHint) profile.manualHintY * frameHeight else null

        return when {
            kalmanInitialized -> {
                val predX = kalmanX + kalmanVx
                val predY = kalmanY + kalmanVy
                candidates.maxByOrNull { c ->
                    val distancePenalty = (hypot((c.cx - predX).toDouble(), (c.cy - predY).toDouble()) / frameWidth).toFloat()
                    c.confidence - distancePenalty
                }
            }
            hintedX != null && hintedY != null -> {
                candidates.maxByOrNull { c ->
                    val distancePenalty = (hypot((c.cx - hintedX).toDouble(), (c.cy - hintedY).toDouble()) / frameWidth).toFloat()
                    c.confidence - distancePenalty * 0.5f
                }
            }
            else -> candidates.maxByOrNull { it.confidence }
        }
    }

    private fun collectComponent(
        seedX: Int,
        seedY: Int,
        visited: BooleanArray,
        minArea: Float,
        maxArea: Float
    ): Candidate? {
        val stack = ArrayDeque<Int>()
        val seed = seedY * downWidth + seedX
        stack.add(seed)
        visited[seed] = true

        var pixelCount = 0
        var edgeCount = 0
        var sumX = 0.0
        var sumY = 0.0
        var weightedX = 0.0
        var weightedY = 0.0
        var weightSum = 0.0
        var lumaSum = 0.0
        var bgSum = 0.0

        var minX = seedX
        var maxX = seedX
        var minY = seedY
        var maxY = seedY

        while (stack.isNotEmpty()) {
            val idx = stack.removeLast()
            val x = idx % downWidth
            val y = idx / downWidth
            val luma = currentLuma[idx].toDouble()
            val bg = backgroundLuma[idx].toDouble()

            pixelCount++
            sumX += x
            sumY += y
            weightedX += x * luma
            weightedY += y * luma
            weightSum += luma
            lumaSum += luma
            bgSum += bg
            minX = min(minX, x)
            maxX = max(maxX, x)
            minY = min(minY, y)
            maxY = max(maxY, y)

            var exposedSides = 0
            for (ny in y - 1..y + 1) {
                for (nx in x - 1..x + 1) {
                    if (nx == x && ny == y) continue
                    if (nx !in 0 until downWidth || ny !in 0 until downHeight) {
                        exposedSides++
                        continue
                    }
                    val nIdx = ny * downWidth + nx
                    val active = brightMask[nIdx] && motionMask[nIdx]
                    if (!active) exposedSides++
                    if (!visited[nIdx] && active) {
                        visited[nIdx] = true
                        stack.add(nIdx)
                    }
                }
            }
            if (exposedSides > 0) edgeCount += exposedSides
        }

        val area = pixelCount.toFloat()
        if (area !in minArea..maxArea) return null

        val centerXDown = if (weightSum > 0.0) (weightedX / weightSum).toFloat() else (sumX / pixelCount).toFloat()
        val centerYDown = if (weightSum > 0.0) (weightedY / weightSum).toFloat() else (sumY / pixelCount).toFloat()
        val radiusDown = sqrt((area / PI).toDouble()).toFloat()

        val perimeter = max(1f, edgeCount / 2f)
        val circularity = ((4f * PI.toFloat() * area) / (perimeter * perimeter)).coerceIn(0f, 1.2f)
        if (circularity < profile.circularityThreshold * 0.85f) return null

        val bboxW = maxX - minX + 1
        val bboxH = maxY - minY + 1
        val aspectPenalty = abs((bboxW.toFloat() / max(1, bboxH)) - 1f)

        val contrast = ((lumaSum / pixelCount) - (bgSum / pixelCount)).toFloat().coerceAtLeast(0f)
        val contrastScore = (contrast / 40f).coerceIn(0f, 1f)
        val roundnessScore = (1f - min(1f, aspectPenalty + abs(circularity - 1f) * 0.7f)).coerceIn(0f, 1f)

        var motionFraction = 0f
        for (y in minY..maxY) {
            for (x in minX..maxX) {
                val idx = y * downWidth + x
                if (brightMask[idx] && motionMask[idx]) motionFraction += 1f
            }
        }
        val motionScore = (motionFraction / max(1f, bboxW * bboxH.toFloat())).coerceIn(0f, 1f)

        val confidence = (roundnessScore * 0.45f + contrastScore * 0.35f + motionScore * 0.20f).coerceIn(0f, 1f)

        val fullX = centerXDown * DOWNSAMPLE
        val fullY = centerYDown * DOWNSAMPLE
        val fullRadius = radiusDown * DOWNSAMPLE

        return Candidate(fullX, fullY, fullRadius, area * DOWNSAMPLE * DOWNSAMPLE, circularity, contrast, confidence)
    }

    private fun kalmanUpdate(measX: Float, measY: Float): PointF {
        if (!kalmanInitialized) {
            kalmanX = measX
            kalmanY = measY
            kalmanVx = 0f
            kalmanVy = 0f
            kalmanP = Array(4) { i -> FloatArray(4) { j -> if (i == j) 100f else 0f } }
            kalmanInitialized = true
            return PointF(measX, measY)
        }

        val pX = kalmanX + kalmanVx
        val pY = kalmanY + kalmanVy

        kalmanP[0][0] += processNoise
        kalmanP[1][1] += processNoise
        kalmanP[2][2] += processNoise * 0.4f
        kalmanP[3][3] += processNoise * 0.4f

        val kx = kalmanP[0][0] / (kalmanP[0][0] + measureNoise)
        val ky = kalmanP[1][1] / (kalmanP[1][1] + measureNoise)
        val kvx = kalmanP[2][2] / (kalmanP[2][2] + measureNoise * 1.8f)
        val kvy = kalmanP[3][3] / (kalmanP[3][3] + measureNoise * 1.8f)

        val dx = measX - pX
        val dy = measY - pY

        kalmanX = pX + kx * dx
        kalmanY = pY + ky * dy
        kalmanVx += kvx * dx
        kalmanVy += kvy * dy

        kalmanP[0][0] *= (1f - kx)
        kalmanP[1][1] *= (1f - ky)
        kalmanP[2][2] *= (1f - kvx)
        kalmanP[3][3] *= (1f - kvy)

        return PointF(kalmanX, kalmanY)
    }

    private fun checkForSwing() {
        val history = synchronized(ballHistory) { ballHistory.toList() }
        if (history.size < 4) return

        val latest = history.last()
        val compare = history[history.size - 4]
        val dtMs = (latest.timestamp - compare.timestamp).toFloat()
        if (dtMs <= 0f) return

        val dx = latest.x - compare.x
        val dy = latest.y - compare.y
        val speedPxPerFrame = (sqrt(dx * dx + dy * dy) / dtMs) * (1000f / TARGET_FPS)

        motionFrameCount = if (speedPxPerFrame >= profile.swingMotionThresholdPx) motionFrameCount + 1 else 0

        if (motionFrameCount >= profile.swingConfirmFrames && !_swingDetected.value) {
            _swingDetected.value = true
            _trackingState.value = TrackingState.SwingInProgress
            _shotSnapshot.value = buildShotSnapshot(history)
        }
    }

    private fun buildShotSnapshot(history: List<BallPosition>): ShotDataSnapshot {
        val frameWidth = lastFrameWidth.toDouble()
        val yardsPerPx = (2.0 * (profile.cameraDistanceFeet / 3.0) * tan(Math.toRadians(CAMERA_HFOV_DEG / 2.0))) / frameWidth
        val feetPerPx = yardsPerPx * 3.0

        val window = history.takeLast(VELOCITY_WINDOW)
        if (window.size < 3) return ShotDataSnapshot.empty(profile)

        val first = window.first()
        val last = window.last()
        val dtSec = (last.timestamp - first.timestamp) / 1000.0
        if (dtSec <= 0.0) return ShotDataSnapshot.empty(profile)

        val dxPx = last.x - first.x
        val dyPx = last.y - first.y

        val vx = (dxPx * feetPerPx) / dtSec
        val vy = -(dyPx * feetPerPx) / dtSec
        val speedFps = hypot(vx, vy)
        val ballSpeedMph = (speedFps / 1.46667).coerceIn(8.0, 230.0)
        val launchAngleDeg = Math.toDegrees(atan2(vy, abs(vx))).coerceIn(0.0, 65.0)

        var curvatureAccum = 0.0
        var curvatureSamples = 0
        for (i in 1 until window.lastIndex) {
            val p0 = window[i - 1]
            val p1 = window[i]
            val p2 = window[i + 1]

            val a = hypot((p1.x - p0.x).toDouble(), (p1.y - p0.y).toDouble())
            val b = hypot((p2.x - p1.x).toDouble(), (p2.y - p1.y).toDouble())
            val c = hypot((p2.x - p0.x).toDouble(), (p2.y - p0.y).toDouble())
            if (a < 0.001 || b < 0.001 || c < 0.001) continue

            val area2 = abs((p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x)).toDouble()
            val curvature = (2.0 * area2) / (a * b * c)
            val turn = sign(((p1.x - p0.x) * (p2.y - p1.y) - (p1.y - p0.y) * (p2.x - p1.x)).toDouble())
            curvatureAccum += curvature * turn
            curvatureSamples++
        }

        val avgCurvature = if (curvatureSamples > 0) curvatureAccum / curvatureSamples else 0.0
        val spinScale = (ballSpeedMph / 160.0).coerceIn(0.35, 1.45)

        val backspinRpm = (1800.0 + launchAngleDeg * 70.0 + speedFps.pow(0.9) * 6.0).coerceIn(1200.0, 8500.0)
        val sidespinRpm = (avgCurvature * 65000.0 * spinScale).coerceIn(-3500.0, 3500.0)

        val swingPathDeg = if (abs(dxPx) > 1f) Math.toDegrees(atan2(-dyPx.toDouble(), dxPx.toDouble())) else 0.0
        val faceAngleDeg = (swingPathDeg * 0.6 + (sidespinRpm / 500.0)).coerceIn(-12.0, 12.0)

        val detectionRate = (history.size.toFloat() / FRAME_HISTORY_SIZE.toFloat() * 100f).coerceIn(0f, 100f)

        return ShotDataSnapshot(
            ballSpeedMph = ballSpeedMph,
            launchAngleDeg = launchAngleDeg,
            swingPathDeg = swingPathDeg,
            faceAngleDeg = faceAngleDeg,
            backspinRpm = backspinRpm,
            sidespinRpm = sidespinRpm,
            estimatedCarryYards = 0.0,
            yardsPerPixel = yardsPerPx,
            frameCount = history.size,
            detectionRatePct = detectionRate,
            rawPositions = history,
            profile = profile
        )
    }

    private fun updateFps(now: Long) {
        frameTimestamps.addLast(now)
        while (frameTimestamps.isNotEmpty() && now - frameTimestamps.first() > 1000L) {
            frameTimestamps.removeFirst()
        }
        _fps.value = frameTimestamps.size.toFloat()
    }
}

data class ShotDataSnapshot(
    val ballSpeedMph: Double,
    val launchAngleDeg: Double,
    val swingPathDeg: Double,
    val faceAngleDeg: Double,
    val backspinRpm: Double,
    val sidespinRpm: Double,
    val estimatedCarryYards: Double,
    val yardsPerPixel: Double,
    val frameCount: Int,
    val detectionRatePct: Float,
    val rawPositions: List<BallPosition>,
    val profile: CalibrationProfile
) {
    companion object {
        fun empty(profile: CalibrationProfile) = ShotDataSnapshot(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            0f,
            emptyList(),
            profile
        )
    }
}
