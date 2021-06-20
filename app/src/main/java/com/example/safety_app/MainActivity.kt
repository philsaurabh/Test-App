package com.example.safety_app

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import com.example.safety_app.ml.StudentTflite
import org.tensorflow.lite.DataType
import org.tensorflow.lite.schema.TensorType.FLOAT16
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {

    lateinit var bitmap: Bitmap
    lateinit var imgview: ImageView




    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imgview = findViewById(R.id.imageView)
        val tv: TextView = findViewById(R.id.textView)
        val select: Button=findViewById(R.id.button)
        select.setOnClickListener(View.OnClickListener {


            var intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.type="image/*"
            startActivityForResult(intent,100)

        })
        val predict:Button=findViewById(R.id.button2)
        predict.setOnClickListener(View.OnClickListener {
            val resized: Bitmap=Bitmap.createScaledBitmap(bitmap,640,640,true)
            val model = StudentTflite.newInstance(this)


// Creates inputs for reference.
            val buffer = TensorImage.fromBitmap(resized)
            val byteBuffer = buffer.buffer
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 320, 320, 3), DataType.FLOAT32)
            inputFeature0.loadBuffer(byteBuffer)

// Runs model inference and gets result.
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer

            tv.setText(outputFeature0.floatArray[0].toString())

// Releases model resources if no longer used.
            model.close()

        })
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        imgview.setImageURI(data?.data)
        val uri = data?.data

        bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)

    }

}