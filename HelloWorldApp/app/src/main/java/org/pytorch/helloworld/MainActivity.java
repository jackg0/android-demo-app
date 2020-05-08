package org.pytorch.helloworld;

import android.content.Context;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import org.apache.commons.math3.complex.Quaternion;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.DoubleBuffer;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity {

  @RequiresApi(api = Build.VERSION_CODES.N)
  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    int numSamples = 0;
    Module module = null;
    BufferedReader csvReader = null;


    try {
      module = Module.load(assetFilePath(this, "script_model.pt"));
      csvReader = new BufferedReader(new FileReader(assetFilePath(this, "calibration.csv")));
    } catch (IOException e) {
      Log.e("PytorchHelloWorld", "Error reading assets", e);
      finish();
    }

    TextView textView = findViewById(R.id.text);
    long startTime = System.nanoTime();
    final long[] shape = {5, 1, 35};

    long[] outputShape = null;

    DoubleBuffer db = Tensor.allocateDoubleBuffer(5 * 35);
    String row = "";
    try {
      while ((row = csvReader.readLine()) != null) {
        numSamples++;
        double data[] = Arrays.stream(row.split(","))
                .mapToDouble(Double::parseDouble)
                .toArray();


        Quaternion G_sternum = new Quaternion(data[0], Arrays.copyOfRange(data, 1, 4));
        Quaternion sternum_G = G_sternum.getConjugate();

        Quaternion G_rightWrist = new Quaternion(data[4], Arrays.copyOfRange(data, 5, 8));
        Quaternion G_leftWrist = new Quaternion(data[8], Arrays.copyOfRange(data, 9, 12));

        Quaternion sternumAcc = new Quaternion(0, Arrays.copyOfRange(data, 12, 15));
        Quaternion rightWristAcc = new Quaternion(0, Arrays.copyOfRange(data, 15, 18));
        Quaternion leftWristAcc = new Quaternion(0, Arrays.copyOfRange(data, 18, 21));

        double sternum_rightWrist[] = quaternionToDouble(sternum_G.multiply(G_rightWrist));
        double sternum_leftWrist[] = quaternionToDouble(sternum_G.multiply(G_leftWrist));
        double sternum_rightWristAcc[] = G_sternum.multiply((rightWristAcc.subtract(sternumAcc)).multiply(sternum_G)).getVectorPart();
        double sternum_leftWristAcc[] = G_sternum.multiply((leftWristAcc.subtract(sternumAcc)).multiply(sternum_G)).getVectorPart();

        db = db.put(sternum_rightWrist);
        db = db.put(sternum_leftWrist);
        db = db.put(sternum_rightWristAcc);
        db = db.put(sternum_leftWristAcc);

        if (numSamples % 5 == 0) {
          final Tensor inputTensor = Tensor.fromBlob(db, shape);

          final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

          outputShape = outputTensor.shape();

          db.clear();
        }
        Thread.sleep(30);
      }

    } catch (Throwable e) {
      e.printStackTrace();
    } finally {
      if (csvReader != null) {
        try {
          csvReader.close();
        } catch (IOException e) {
          e.printStackTrace();
        }
      }
    }

    long endTime = System.nanoTime();
    long nanoToMilli = 1000000;
    long elapsedTime = (endTime - startTime) / (nanoToMilli);
    textView.setText("Time to run " + Integer.toString(numSamples) + " samples: " + Long.toString(elapsedTime) + " milliseconds");
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

  public double[] quaternionToDouble(Quaternion q)
  {
    return new double[] { q.getQ0(), q.getQ1(), q.getQ2(), q.getQ3() };
  }
}
