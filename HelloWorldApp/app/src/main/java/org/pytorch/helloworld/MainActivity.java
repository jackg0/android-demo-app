package org.pytorch.helloworld;

import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends AppCompatActivity {

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    Module module = null;
    try {
      module = Module.load(assetFilePath(this, "script_model.pt"));
    } catch (IOException e) {
      Log.e("PytorchHelloWorld", "Error reading assets", e);
      finish();
    }

    TextView textView = findViewById(R.id.text);
    long startTime = System.nanoTime();
    int numSamples = 10000;
    for (int i = 0; i < numSamples; i++)
    {
      final long[] shape = {35};
      final Tensor inputTensor = Tensor.fromBlob(Tensor.allocateDoubleBuffer(35), shape);

      final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
    }
    long endTime = System.nanoTime();
    long nanoToMilli = 1000000;
    long elapsedTime = (endTime - startTime) / (nanoToMilli);
    textView.setText("Time to run " + Integer.toString(numSamples) + " samples: " + Long.toString(elapsedTime) + " milliseconds" );

  }

  /**
   * Copies specified asset to the file in /files app directory and returns this file absolute path.
   *
   * @return absolute file path
   */
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
}
