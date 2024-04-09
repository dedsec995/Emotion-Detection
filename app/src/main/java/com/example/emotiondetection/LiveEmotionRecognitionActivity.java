package com.example.emotiondetection;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.YuvImage;
import android.media.Image;
import android.util.Log;
import android.util.Pair;
import android.util.SparseArray;
import android.view.TextureView;
import android.view.ViewStub;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import androidx.camera.core.ImageProxy;

import com.google.android.gms.vision.Frame;
import com.google.android.gms.vision.face.Face;
import com.google.android.gms.vision.face.FaceDetector;

import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Objects;


public class LiveEmotionRecognitionActivity extends com.example.emotiondetection.AbstractCameraXActivity<LiveEmotionRecognitionActivity.AnalysisResult> {
    private Module mModule = null;
    private TextView mResultView;

    private final static int ANGRY = 0;
    private final static int DISGUST = 1;
    private final static int FEAR = 2;
    private final static int HAPPY = 3;
    private final static int NEUTRAL = 4;
    private final static int SAD = 5;
    private final static int SUPRISE = 6;
    private Bitmap croppedBitmap;



    static class AnalysisResult {
        private final String mResults;

        public AnalysisResult(String results) {
            mResults = results;
        }
    }

    @Override
    protected int getContentViewLayoutId() {
        return R.layout.activity_live_emotion_recognition;
    }

    @Override
    protected TextureView getCameraPreviewTextureView() {
        mResultView = findViewById(R.id.resultView);
        return ((ViewStub) findViewById(R.id.emotion_recognition_texture_view_stub))
                .inflate()
                .findViewById(R.id.object_detection_texture_view);
    }

    @Override
    protected void applyToUiAnalyzeImageResult(AnalysisResult result) {
        mResultView.setText(result.mResults);
        mResultView.invalidate();
    }

    private Bitmap imgToBitmap(Image image) {
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);

        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }


    @Override
    @WorkerThread
    @Nullable
    protected AnalysisResult analyzeImage(ImageProxy image, int rotationDegrees) {
        if (mModule == null) {
            try {
                mModule = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "mobilenet_v2_lite2.ptl"));
            } catch (IOException e) {
                return null;
            }
        }

        Bitmap bitmap = detectFacesAndCrop(imgToBitmap(Objects.requireNonNull(image.getImage())));
//        Bitmap bitmap = imgToBitmap(Objects.requireNonNull(image.getImage()));
        Matrix matrix = new Matrix();
        matrix.postRotate(90.0f);
        bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        bitmap = Bitmap.createScaledBitmap(bitmap, MainActivity.SIZE, MainActivity.SIZE, true);

        Pair<Integer, Long> idxTm = MainActivity.bitmapRecognition(bitmap, mModule);
        int maxScoreIdx = idxTm.first;
        long inferenceTime = idxTm.second;

        String result = getEmotionLabel(maxScoreIdx);
        Log.d("MyTag", "analyzeImage result: " + result);
        Log.d("MyTag", "analyzeImage maxScoreIdx: " + maxScoreIdx);
        return new AnalysisResult(String.format("%s - %dms", result, inferenceTime));
    }

    private String getEmotionLabel(int maxScoreIdx) {
        switch (maxScoreIdx) {
            case ANGRY:
                return "ANGRY";
            case DISGUST:
                return "DISGUST";
            case FEAR:
                return "FEAR";
            case HAPPY:
                return "HAPPY";
            case NEUTRAL:
                return "NEUTRAL";
            case SAD:
                return "SAD";
            case SUPRISE:
                return "SUPRISE";
            default:
                return String.valueOf((char)(1 + maxScoreIdx + 64));
        }
    }

    private Bitmap detectFacesAndCrop(Bitmap bitmap) {
        FaceDetector faceDetector = new FaceDetector.Builder(getApplicationContext())
                .setTrackingEnabled(false)
                .setLandmarkType(FaceDetector.ALL_LANDMARKS)
                .build();

        if (!faceDetector.isOperational()) {
            // Handle the error
            return bitmap;
        }

        Frame frame = new Frame.Builder().setBitmap(bitmap).build();
        SparseArray<Face> faces = faceDetector.detect(frame);

        if (faces.size() == 0) {
            // No faces detected
            return bitmap;
        }

        // Crop the bitmap to the first face detected
        Face face = faces.valueAt(0);
        float x1 = face.getPosition().x;
        float y1 = face.getPosition().y;
        float x2 = x1 + face.getWidth();
        float y2 = y1 + face.getHeight();

        RectF rectF = new RectF(x1, y1, x2, y2);
        croppedBitmap = Bitmap.createBitmap(bitmap,
                (int) rectF.left,
                (int) rectF.top,
                (int) rectF.width(),
                (int) rectF.height());
        return croppedBitmap;
    }
}