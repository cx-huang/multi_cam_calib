#pragma once
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <time.h>
#include <typeinfo.h>

using namespace std;
using namespace cv;

class CalibBall
{
public:
	int _cam_num;
	vector<int> _cam_list;
	string _sample_filepath;
	vector<string> _imagelist_vector;
	vector<string> _backgroundlist_vector;
	float _ball_radii;
	float _tag_pnt_dist;
	vector<int> _principal_pnt_x;
	vector<int> _principal_pnt_y;
	vector<int> _focal_len;
	CalibBall(string config_filename);
	~CalibBall();
	void Run(string config_filename);
private:
	void RunOnce(vector<string> imagelist, vector<string> backgroundlist, vector<Point3d> &cur_pntsW, vector<vector<Point2d>> &cur_pntsI, vector<vector<int>> &cur_visibility, vector<Mat> &R, vector<Mat> &T);
	void CalR(vector<Mat> pntpairs_src, vector<Mat> &pntpairs_dst, vector<int> &visibility_src, vector<int> &visibility_dst, vector<Point3d> &pntsW, Mat &R);
	void FindCircle(Mat &ref, Mat background, Vec3d &circle_, int cam_idx);
	void FindPoints(Mat image, vector<vector<Point2f>> &pntpairsI, Vec3d circle_, int cam_idx);
	void PointI2W(vector<vector<Point2f>> pntpairsI, vector<vector<Point3d>> &pntpairsW, Vec3d circle_, Mat T, int cam_idx);
	void CalT(Mat &T, Vec3d circle_, int cam_idx);
	void NormalizePoints(vector<vector<Point3d>> pntpairW, vector<Mat> &pntpairW_normed);
	void ProcessForSBA(vector<vector<Point2f>> pntpairsI, vector<Mat> pntpairsW_normed, vector<Point2d> &pntsI, vector<Point3d> pntsW, vector<int> &visibility);
	void ProjectToImg(vector<string> &imagelist, vector<Mat> cam_matrices, vector<Mat> R, vector<Mat> T, vector<Point3d> pntsW, vector<vector<Point2d>> pntsI, vector<vector<int>> visibility);
	void ScaleToWorld(vector<Point3d> &pntsW, vector<Mat> &T);
};

void DetectCircle(Mat image, Vec3d &circle_, double min_circle_radius, double max_circle_radius);
void TwoPass(const Mat &binary_image, Mat &label_image);
