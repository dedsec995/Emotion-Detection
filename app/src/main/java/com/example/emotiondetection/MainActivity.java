package com.example.emotiondetection;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Pair;
import android.view.Menu;
import android.view.MenuItem;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.FloatBuffer;

public class MainActivity extends AppCompatActivity {

    public final static int SIZE = 224;

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    public static Pair<Integer, Long> bitmapRecognition(Bitmap bitmap, Module module) {


        FloatBuffer inTensorBuffer = Tensor.allocateFloatBuffer(3 * SIZE * SIZE);
        for (int x = 0; x < SIZE; x++) {
            for (int y = 0; y < SIZE; y++) {
                int colour = bitmap.getPixel(x, y);

                int red = Color.red(colour);
                int blue = Color.blue(colour);
                int green = Color.green(colour);
                inTensorBuffer.put(x + SIZE * y, (float) blue);
                inTensorBuffer.put(SIZE * SIZE + x + SIZE * y, (float) green);
                inTensorBuffer.put(2 * SIZE * SIZE + x + SIZE * y, (float) red);
            }
        }

        Tensor inputTensor = Tensor.fromBlob(inTensorBuffer, new long[]{1, 3, SIZE, SIZE});

        final long startTime = SystemClock.elapsedRealtime();
        Tensor outTensor = module.forward(IValue.from(inputTensor)).toTensor();
        final long inferenceTime = SystemClock.elapsedRealtime() - startTime;

        final float[] scores = outTensor.getDataAsFloatArray();
        float maxScore = -Float.MAX_VALUE;
        int maxScoreIdx = -1;
        for (int i = 0; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                maxScoreIdx = i;
            }
        }
        return new Pair<>(maxScoreIdx, inferenceTime);
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

}