package com.example.emotiondetection;


import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.util.Pair;
import android.util.SparseArray;
import android.view.TextureView;
import android.view.ViewStub;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
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
import java.io.OutputStream;
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
    private final static int SURPRISE = 6;



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

    private Bitmap imgToBitmap(Image image, int rotationDegrees) {
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
        Bitmap bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
        if (rotationDegrees != 0) {
            Matrix matrix = new Matrix();
            matrix.postRotate(rotationDegrees);
            bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        }
        return bitmap;
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
        Pair<Boolean,Bitmap> detectionResult = detectAndCropFace(getApplicationContext(),imgToBitmap(image.getImage(),rotationDegrees));
        boolean faceDetected = detectionResult.first;
        if (faceDetected){
            Bitmap bitmap = detectionResult.second;
            Matrix matrix = new Matrix();
            bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
            bitmap = Bitmap.createScaledBitmap(bitmap, MainActivity.SIZE, MainActivity.SIZE, true);
            Pair<Integer, Long> idxTm = MainActivity.bitmapRecognition(bitmap, mModule);
            int maxScoreIdx = idxTm.first;
            long inferenceTime = idxTm.second;
            String result = getEmotionLabel(maxScoreIdx);
            return new AnalysisResult(String.format("%s - %dms", result, inferenceTime));
        }
        else{
            return new AnalysisResult(String.format("%s - %dms", "No face Detected", 0));
        }


//        try {
//            saveBitmap(getApplicationContext(),bitmap,Bitmap.CompressFormat.PNG, "image/png", "bitmap");
//            Toast.makeText(getApplicationContext(), "Demo Saved!", Toast.LENGTH_SHORT).show();
//        } catch (IOException e) {
//            Toast.makeText(getApplicationContext(), "Demo not Saved!", Toast.LENGTH_SHORT).show();
//            throw new RuntimeException(e);
//        }

//        Log.d("MyTag", "analyzeImage result: " + result);
//        Log.d("MyTag", "analyzeImage maxScoreIdx: " + maxScoreIdx);

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
            case SURPRISE:
                return "SUPRISE";
            default:
                return String.valueOf((char)(1 + maxScoreIdx + 64));
        }
    }


    public static void saveBitmap(@NonNull final Context context, @NonNull final Bitmap bitmap,
                                  @NonNull final Bitmap.CompressFormat format,
                                  @NonNull final String mimeType,
                                  @NonNull final String displayName) throws IOException {

        final ContentValues values = new ContentValues();
        values.put(MediaStore.MediaColumns.DISPLAY_NAME, displayName);
        values.put(MediaStore.MediaColumns.MIME_TYPE, mimeType);
        values.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DCIM);

        final ContentResolver resolver = context.getContentResolver();
        Uri uri = null;

        try {
            final Uri contentUri = MediaStore.Images.Media.EXTERNAL_CONTENT_URI;
            uri = resolver.insert(contentUri, values);

            if (uri == null)
                throw new IOException("Failed to create new MediaStore record.");

            try (final OutputStream stream = resolver.openOutputStream(uri)) {
                if (stream == null)
                    throw new IOException("Failed to open output stream.");

                if (!bitmap.compress(format, 95, stream))
                    throw new IOException("Failed to save bitmap.");
            }

        }
        catch (IOException e) {
            if (uri != null) {
                // Don't leave an orphan entry in the MediaStore
                resolver.delete(uri, null, null);
            }
            throw e;
        }
    }

    public static Pair<Boolean,Bitmap> detectAndCropFace(Context context, Bitmap originalBitmap) {
        // Initialize the face detector
        FaceDetector faceDetector = new FaceDetector.Builder(context)
                .setTrackingEnabled(false)
                .setLandmarkType(FaceDetector.ALL_LANDMARKS)
                .build();

        // Build the frame
        Frame frame = new Frame.Builder().setBitmap(originalBitmap).build();

        // Detect faces
        SparseArray<Face> faces = faceDetector.detect(frame);

        // Check if any face is detected
        if (faces.size() == 0) {
//            Toast.makeText(context, "Face not Detected!", Toast.LENGTH_SHORT).show();
            // No faces detected, return original bitmap
            return new Pair<>(false,originalBitmap);
        }

        // Get the first face detected
        Face face = faces.valueAt(0);

        // Calculate the position of the face
        float x1 = face.getPosition().x;
        float y1 = face.getPosition().y;
        float x2 = x1 + face.getWidth();
        float y2 = y1 + face.getHeight();

        // Crop the face from the original bitmap
        Bitmap croppedBitmap = Bitmap.createBitmap(originalBitmap, (int) x1, (int) y1, (int) (x2 - x1), (int) (y2 - y1));
        Matrix matrix = new Matrix();
        croppedBitmap = Bitmap.createBitmap(croppedBitmap, 0, 0, croppedBitmap.getWidth(), croppedBitmap.getHeight(), matrix, true);
//        Toast.makeText(context, "Face Detected at: " + x1 + y1, Toast.LENGTH_SHORT).show();

        // Release the face detector
        faceDetector.release();

        return new Pair<>(true,croppedBitmap);
    }
}