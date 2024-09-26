package com.example.imageproc;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.audiofx.AudioEffect;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.util.Log;
import android.widget.TextView;

import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.android.JavaCameraView;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.features2d.BFMatcher;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.ORB;
import org.opencv.features2d.SIFT;
import org.opencv.imgproc.Imgproc;
import org.opencv.features2d.FastFeatureDetector;
import org.opencv.features2d.Feature2D;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.BOWImgDescriptorExtractor;
import org.opencv.features2d.BRISK;
import org.opencv.utils.Converters;


import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.regex.Matcher;

public class MainActivity extends CameraActivity {

    CameraBridgeViewBase cameraBridgeViewBase;
    CameraBridgeViewBase.CvCameraViewFrame prevFrame;
    TextView tv;
    ImageView matResult;
    double rotationSum;
    //double movement;

    Mat curr_gray , prev_gray , dst , descriptor1 , descriptor2 ,result;
    boolean is_init;
    MatOfKeyPoint keyPoints1 , keyPoints2 ;
    MatOfDMatch matches;
    //List<MatOfPoint> keyPoints1,keyPoints2;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        getPermisson();
        is_init=false;

        cameraBridgeViewBase = findViewById( R.id.cameraView );
        tv = findViewById( R.id.textView );
        matResult = findViewById( R.id.matchResult );

        cameraBridgeViewBase.setCvCameraViewListener(new CameraBridgeViewBase.CvCameraViewListener2() {
            @Override
            public void onCameraViewStarted(int width, int height) {
                System.out.print("hello");
                curr_gray = new Mat();
                prev_gray = new Mat();

                dst = new Mat();
                result = new Mat();

                keyPoints1 = new MatOfKeyPoint();
                keyPoints2 = new MatOfKeyPoint();

                descriptor1 = new Mat();
                descriptor2 = new Mat();

                matches = new MatOfDMatch();

                rotationSum = 0;
                //cnts = new ArrayList<>();
            }

            @Override
            public void onCameraViewStopped() {}

            @Override
            public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {//定義每偵的工作

                if( !is_init ){

                    prevFrame = inputFrame;
                    prev_gray = prevFrame.gray();
                    is_init = true;
                    return prev_gray;//剛開始沒初始 先回傳當前畫面灰度

                }

                curr_gray = inputFrame.gray();

                ORB orb = ORB.create();
                orb.setEdgeThreshold( 40 );//設置找keypoint時圖像邊緣的閥值
                orb.setFastThreshold( 70 );//檢測keypoint時的閥值

                keyPoints1.release();
                keyPoints2.release();

                descriptor1.release();
                descriptor2.release();

                matches.release();
                dst.release();

                orb.detectAndCompute( prev_gray , new Mat() , keyPoints1 , descriptor1 );//尋找並計算keypoint的descriptor
                orb.detectAndCompute( curr_gray , new Mat() , keyPoints2 , descriptor2 );

                BFMatcher matcher = BFMatcher.create( Core.NORM_HAMMING , true );
                if( descriptor1.empty() || descriptor2.empty() ){//若兩偵圖其一的關鍵點描述子為空
                    return curr_gray;
                }else{
                    matcher.match( descriptor1 , descriptor2 , matches );//匹配descriptor 並將結果存在matches
                }

                if( matches.empty() ){
                    return curr_gray;
                }

                List<DMatch> matchList = matches.toList();//取前10可靠的匹配結果
                matchList.sort( (a, b) -> Double.compare(a.distance, b.distance) );

                List<DMatch> topMatches = new ArrayList<>();
                if( matchList.size() > 20 ){
                    topMatches =  matchList.subList(0,20);
                }else{
                    topMatches = matchList;
                }

                List<Point> srcPointsList = new ArrayList<>();//第一張圖的匹配點座標
                List<Point> dstPointsList = new ArrayList<>();//第二章圖的匹配點座標
                KeyPoint[] keypoints1Array = keyPoints1.toArray();
                KeyPoint[] keypoints2Array = keyPoints2.toArray();
                for (DMatch match : topMatches) {
                    srcPointsList.add(keypoints1Array[match.queryIdx].pt);
                    dstPointsList.add(keypoints2Array[match.trainIdx].pt);
                }

                Mat srcPoints = Converters.vector_Point2f_to_Mat(srcPointsList);
                Mat dstPoints = Converters.vector_Point2f_to_Mat(dstPointsList);

                // 計算M 旋轉平移矩陣
                Mat M = Calib3d.estimateAffinePartial2D(srcPoints, dstPoints);

                // 提取旋轉角度
                double rotationAngle = -Math.atan2(M.get(0, 1)[0], M.get(0, 0)[0]) * 180 / Math.PI;
                rotationSum += rotationAngle;

                runOnUiThread(new Runnable() {//這裡計算並顯示兩偵圖片中匹配到的keypoint數量 上限500
                    @Override
                    public void run() {

                        tv.setText(String.valueOf(rotationSum));

                    }
                });

                Features2d.drawKeypoints( curr_gray , keyPoints2 , dst );

                prev_gray = curr_gray;

                return dst;

            }
        });



        if( OpenCVLoader.initDebug() ){

            cameraBridgeViewBase.enableView();

        }

    }

    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList( cameraBridgeViewBase );
    }

    @Override
    protected void onResume(){
        super.onResume();
        cameraBridgeViewBase.enableView();
    }

    @Override
    protected void onDestroy(){
        super.onDestroy();
        cameraBridgeViewBase.disableView();
    }

    @Override
    protected void onPause(){
        super.onPause();
        cameraBridgeViewBase.disableView();
    }

    void getPermisson(){
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED ) {
            requestPermissions( new String[]{Manifest.permission.CAMERA} , 101 );
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode , @NonNull String[] permissons , @NonNull int[] grantResults ){

        super.onRequestPermissionsResult( requestCode , permissons , grantResults );
        if( grantResults.length>0 && grantResults[0] != PackageManager.PERMISSION_GRANTED ){//沒得到權限

            getPermisson();

        }

    }

}
