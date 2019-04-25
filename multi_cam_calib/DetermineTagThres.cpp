#include "CalibBall.h"

#define SCALE 10
const int tag_thres_slider_max = 100;
int tag_thres_slider;
extern int tag_thres;
Mat image;

void OnTrackbar(int, void*)
{
	vector<Mat> rgb;
	split(image, rgb);
	Mat tag_mask = rgb[1] - rgb[2];
	tag_thres = tag_thres_slider;
	tag_mask = tag_mask > tag_thres;
	tag_mask = 255 - tag_mask;
	imshow("Determine Tag Threshold", tag_mask);
}

void DetermineTagThres(string image_path)
{
	Mat src_image = imread(image_path);
	resize(src_image, image, Size(), 1.0 / SCALE, 1.0 / SCALE);
	tag_thres_slider = 0;
	namedWindow("Determine Tag Threshold", 0);
	stringstream TrackbarName;
	TrackbarName << "threshold";
	createTrackbar(TrackbarName.str(), "Determine Tag Threshold", &tag_thres_slider, tag_thres_slider_max, OnTrackbar);
	OnTrackbar(tag_thres_slider, 0);
	waitKey(0);
}