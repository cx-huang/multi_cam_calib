#pragma once
#ifndef CCALIBBALL_H
#define CCALIBBALL_H

#define _CRT_SECURE_NO_WARNINGS

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/cvsba.h>
#include <stdio.h>
#include <time.h>
#include <map>
#include <iostream>
#include <assert.h> 
#include <fstream>
using namespace cv;
using namespace std;

#define square(x) ((x)*(x))

class CCalibBall
{
public:
	int cam_num;
	int kinect_num;
	int cam_used_num;
	int code_num;
	string code;
	string kinect_calib_name, camera_calib_name;
	vector<int> cam_list;
	string filepath;
	int NPoints;
	vector<string> backgroundlist;
	vector<string> imagelist_vector;
	double radii;				//大球的直径
	double DF;					//角点间距
	vector<double> focal_length;		//初始焦距
	double init_cx, init_cy;
	string IsRotated;
	int IsAuto;

	CCalibBall(char config_name[]);
	~CCalibBall();
	void run(char config_name[]);
private:
	int run_once(const vector<string> &imagelist, vector< Point3d> &points, vector< vector < Point2d > > &imagePoints, vector< vector< int > > &visibility, vector< Mat > &R, vector< Mat > &T, vector<Mat> cameraMatrix);
	void FindPoints(const vector<string> &imagelist, vector< Mat > &spheres, vector< vector< vector<Point2f> > > &pointsImg, vector< vector< vector<Point3d> > > &points3D, vector< vector <Point2f> > &MarkerPosition, int *first_label, vector<Vec3d> &cicles);
	void FindBigCircle(Mat background, Mat &ref, Mat &BigMarker,Mat &mask_dst,  Vec3d &circle, int idx);
	void FindCorners(Mat diff_image, vector< vector<Point2f> > &pointsImg, Vec3d circle_, int idx);
	void Cal3D(Vec3d circles, Mat &sphere, vector< vector<Point2f> > pointsImg, vector< vector<Point3d> > &points3D, double F);
	void RANSAC(vector< Matx33d > points3D_src, vector< Matx33d > &points3D_dst, Mat &R, vector<Point3d> &point3d_final, vector<int> &visibility_src, vector<int> &visibility_dst);
	void Two_Pass(const Mat& _binImg, Mat& _lableImg);
	void Points3DToMat(vector< vector<Point3d> > points3D, vector<Matx33d> &points);
	void ProcessForSba(vector<vector<Point2f>> pointImg_src, vector<Point2d> &pointImg_dst, vector< Point3d> points, vector< Matx33d > points3D_normed, vector<int> &visibility);
	void OutputParam(vector< Mat > cameraMatrix, vector< Mat > R, vector< Mat > T, vector< Mat > distCoeffs);
	void ProjectToImg(const vector<string> &imagelist, vector< Mat > cameraMatrix, vector< Mat > R, vector< Mat > T, vector< Point3d> points, vector<vector< Point2d >> imagePoints, vector<vector<int>> visibility);
	void ScaleToWorld(vector< Point3d> &points, vector< Mat > &T);	
};
void DetectCircle(Mat image, Vec3d &circle_, double minCircleRadius, double maxCircleRadius);
void AutoThres(Mat src, Mat &dst);
void imfillholes(Mat src, Mat &dst);
#endif
