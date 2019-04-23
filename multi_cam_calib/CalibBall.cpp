#include "CalibBall.h"
#include "opencv2/cvsba.h"

#define IS_DEBUG
#define IS_OUTPUT_CIRCLE
#define IS_DRAW_POINTS
#define IS_PROJ
#define MINR 1000	//in FindCircle(): the minimum radius of the circle in the image
#define MAXR 2000	//in FindCircle(): the maximum raidus of the circle in the image
#define TAG_THRES 50//in FindPoints(): for binarization of tag mask
#define SQUARE(x) ((x) * (x))
#define MAX(a, b) ((a>b)?(a):(b))
#define ASSERT(expression)	if(expression)\
							{\
								NULL; \
							}\
							else\
							{\
								cout << "[Assertion failed!] file # " << __FILE__ << " #, line # " << __LINE__ << " #" << endl; \
								abort();\
							}

CalibBall::CalibBall(
	string config_filename
)
{
	FileStorage fp(config_filename, FileStorage::READ);
	if (fp.isOpened() == false)
	{
		std::cout << "cannot open file # " << config_filename << " #" << endl;
		exit(-1);
	}

	fp["cam_num"] >> _cam_num;
	fp["cam_list"] >> _cam_list;
	fp["sample_filepath"] >> _sample_filepath;
	fp["imagelist_vector"] >> _imagelist_vector;
	fp["backgroundlist_vector"] >> _backgroundlist_vector;
	fp["ball_radii"] >> _ball_radii;
	fp["tag_pnt_dist"] >> _tag_pnt_dist;
	fp["principal_pnt_x"] >> _principal_pnt_x;
	fp["principal_pnt_y"] >> _principal_pnt_y;
	fp["focal_len"] >> _focal_len;
	
	ASSERT(_cam_num > 0 && _cam_num == _cam_list.size());
	ASSERT(!_imagelist_vector.empty() && _imagelist_vector.size() == _backgroundlist_vector.size());
	ASSERT(_cam_num == _principal_pnt_x.size() && _cam_num == _principal_pnt_y.size() && _cam_num == _focal_len.size());
}

CalibBall::~CalibBall()
{

}

void CalibBall::Run(
	string config_filename
)
{
	clock_t start = clock();
	FileStorage fp(config_filename, FileStorage::READ);
	if (fp.isOpened() == false)
	{
		std::cout << "cannot open file # " << config_filename << " #" << endl;
		exit(-1);
	}

	vector<Mat> cam_matrices(_cam_num), R(_cam_num), T(_cam_num), dis_coeffs(_cam_num);
	for (int i = 0; i < _cam_num; ++i)
	{
		cam_matrices[i] = Mat::zeros(3, 3, CV_64FC1);
		cam_matrices[i].at<double>(0, 0) = _focal_len[_cam_list[i]];
		cam_matrices[i].at<double>(1, 1) = _focal_len[_cam_list[i]];
		cam_matrices[i].at<double>(0, 2) = _principal_pnt_x[_cam_list[i]];
		cam_matrices[i].at<double>(1, 2) = _principal_pnt_y[_cam_list[i]];
		cam_matrices[i].at<double>(2, 2) = 1;
		dis_coeffs[i] = Mat::zeros(5, 1, CV_64FC1);
	}

	vector<Point3d> pntsW;
	vector<vector<Point2d>> pntsI(_cam_num);
	vector<vector<int>> visibility(_cam_num);
	vector<string> cur_imagelist, cur_backgroundlist;
	for (int i = 0; i < _imagelist_vector.size(); i++)
	{
		cur_imagelist.clear();
		cur_backgroundlist.clear();
		fp[_imagelist_vector[i]] >> cur_imagelist;
		fp[_backgroundlist_vector[i]] >> cur_backgroundlist;
		ASSERT(!cur_imagelist.empty() && cur_imagelist.size() == cur_backgroundlist.size());
		
		vector<Point3d> cur_pntsW;
		vector<vector<Point2d>> cur_pntsI(_cam_num);
		vector<vector<int>> cur_visibility(_cam_num);
		RunOnce(cur_imagelist, cur_backgroundlist, cur_pntsW, cur_pntsI, cur_visibility, R, T);

	}

	clock_t end = clock();
	std::cout << "** Duration **" << endl << (end - start) / CLOCKS_PER_SEC << " s (about " << (end - start) / CLOCKS_PER_SEC / 60 << " min)" << endl;
}

void CalibBall::RunOnce(
	vector<string> imagelist,
	vector<string> backgroundlist,
	vector<Point3d> &cur_pntsW,
	vector<vector<Point2d>> &cur_pntsI,
	vector<vector<int>> &cur_visibility,
	vector<Mat> &R,
	vector<Mat> &T
)
{
	vector<vector<vector<Point3d>>> pntpairsW(_cam_num);
	vector<vector<vector<Point2f>>> pntpairsI(_cam_num);
	vector<Vec3d> circles(_cam_num);
	//step1: find points on the ball and calculate T
#pragma omp parallel for
	for (int i = 0; i < _cam_num; i++)
	{
		Mat image = imread(_sample_filepath + imagelist[i]);
		ASSERT(image.rows != 0 && image.cols != 0);

		Mat background = imread(_sample_filepath + backgroundlist[i]);
		ASSERT(background.rows != 0 && background.cols != 0);

		FindCircle(image, background, circles[i], i);	//2min or so
		FindPoints(image, pntpairsI[i], circles[i], i);	//12min or so

	}
}

void CalibBall::FindCircle(
	Mat &image, 
	Mat background, 
	Vec3d &circle_, 
	int cam_idx
)
{
	Mat gray, BigMarker, mask_dst;
	absdiff(background, image, gray);
	cvtColor(gray, gray, CV_BGR2GRAY);
	GaussianBlur(gray, gray, Size(25, 25), 2, 2);
	gray = gray > 10;
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(20, 20));
	morphologyEx(gray, gray, MORPH_CLOSE, element, Point(-1, -1), 2);
	vector<vector<Point> > contours;
	findContours(gray, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
	double max_area = 0;
	int max_ID = -1;
	for (int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if (area > max_area)
		{
			max_area = area;
			max_ID = i;
		}
	}
	gray.setTo(0);
	drawContours(gray, contours, max_ID, Scalar(255));
	DetectCircle(gray, circle_, MINR, MAXR);
	if (cvRound(circle_[2]) <= 0) {
		printf("circle not detected in %d\n", cam_idx);
		exit(-1);
	}
	Point2i bestCircleCenter(cvRound(circle_[0]), cvRound(circle_[1]));
	Mat mask(image.size(), CV_8UC1, Scalar(0));
	circle(mask, bestCircleCenter, cvRound(circle_[2]) - 120, Scalar(255), CV_FILLED);
	Mat dst;
	image.copyTo(dst, mask);
	mask.setTo(0);
	drawContours(mask, contours, max_ID, Scalar(255), CV_FILLED);
	circle(mask, bestCircleCenter, cvRound(circle_[2]), Scalar(0), CV_FILLED);
	element = getStructuringElement(MORPH_ELLIPSE, Size(10, 10));
	erode(mask, mask, element);
	findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
	max_area = 0;
	max_ID = -1;
	for (int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if (area > max_area)
		{
			max_area = area;
			max_ID = i;
		}
	}
	mask.setTo(0);
	drawContours(mask, contours, max_ID, Scalar(255), CV_FILLED);
	//image.copyTo(BigMarker, mask);
	//cvtColor(BigMarker, BigMarker, CV_BGR2GRAY);
	//mask_dst = mask;
#ifdef IS_OUTPUT_CIRCLE
	stringstream filename;
	filename << "out//contour_" << cam_idx << ".jpg";
	imwrite(filename.str(), gray);
	filename.clear();
	filename.str("");

	Mat ref1 = image.clone();
	circle(ref1, bestCircleCenter, cvRound(circle_[2]), Scalar(255, 255, 255), 5);
	filename << "out//circle_" << cam_idx << ".jpg";
	imwrite(filename.str(), ref1);
	filename.clear();
	filename.str("");
#endif
	image = dst;
#ifdef IS_OUTPUT_CIRCLE
	filename << "out//ref_cut_" << cam_idx << ".jpg";
	imwrite(filename.str(), image);
#endif
	cout << "> " << cam_idx << " FindCircle() done!!" << endl;
}

void CalibBall::FindPoints(
	Mat image, 
	vector<vector<Point2f>> &pntpairsI,
	Vec3d circle_,
	int cam_idx
)
{
	//step1: seperate tags
	vector<Mat> rgb;
	split(image, rgb);
	Mat tag_mask = rgb[1] - rgb[2];
	tag_mask = tag_mask > TAG_THRES;
	tag_mask = 255 - tag_mask;

	//step2: find points on the tags
	Mat gray_image;
	cvtColor(image, gray_image, CV_BGR2GRAY);
	vector<Point2f> pnts;
	goodFeaturesToTrack(gray_image, pnts, 100, 0.1, 60, tag_mask, 5);	//10s or so
	clock_t end_2 = clock();
	vector<Point2f> tag_pnts;
	for (int i = 0; i < pnts.size(); i++)
	{
		if (tag_mask.at<uchar>((int)pnts[i].y, (int)pnts[i].x) == 255)
		{
			tag_pnts.push_back(pnts[i]);
		}
	}
	pnts.clear();

	Mat binary_image, label_image;
	Mat tag_mask_ROI = tag_mask(Rect((int)(circle_[0] - circle_[2]), (int)(circle_[1] - circle_[2]), (int)(2 * circle_[2]), (int)(2 * circle_[2])));
	Mat mask(tag_mask_ROI.size(), CV_8UC1, Scalar(1));
	mask.copyTo(binary_image, tag_mask_ROI);
	TwoPass(binary_image, label_image);	//1min or so

	vector<vector<int>> tag;//tag[i][j] = k: the ith tag's jth point is tag_pnt[k]
	map<int, int> region;	//region[i] = j: the ith region is the jth tag
	int tag_idx = 0;
	for (int i = 0; i < tag_pnts.size(); i++)
	{
		int region_idx = label_image.at<int>(tag_pnts[i].y - (circle_[1] - circle_[2]), tag_pnts[i].x - (circle_[0] - circle_[2]));
		if (region.count(region_idx) <= 0)
		{
			region[region_idx] = tag_idx++;
			vector<int> temp_vec;
			temp_vec.push_back(i);
			tag.push_back(temp_vec);
		}
		else
		{
			tag[region[region_idx]].push_back(i);
		}
	}

	TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001);
	float ratio = 0.1f;	//TODO: figure out a proper value
	for (int i = 0; i < tag.size(); i++)
	{
		if (tag[i].size() != 2)
			continue;
	
		vector<Point2f> pntpair(2);
		pntpair[0] = tag_pnts[tag[i][0]];
		pntpair[1] = tag_pnts[tag[i][1]];
		cornerSubPix(gray_image, pntpair, Size(5, 5), Size(-1, -1), criteria);

		//step4: differentiate left point with right point
		Point2f pnt1 = (1 - ratio) * pntpair[0] + ratio * pntpair[1];
		Point2f pnt2 = (pntpair[1] - pntpair[0]) * ratio;
		Point2f pnt_new1(pnt1.x - pnt2.y, pnt1.y + pnt2.x);
		Point2f pnt_new2(pnt1.x + pnt2.y, pnt1.y - pnt2.x);
		if (gray_image.at<uchar>(pnt_new1) > gray_image.at<uchar>(pnt_new2))
		{
			Point2f temp = pntpair[0];
			pntpair[0] = pntpair[1];
			pntpair[1] = temp;
		}
		//step5: record corners
		pntpairsI.push_back(pntpair);
#ifdef IS_DRAW_POINTS
		circle(gray_image, pntpair[0], 3, Scalar(255), CV_FILLED);
		circle(gray_image, pntpair[1], 3, Scalar(0), CV_FILLED);
#endif
	}
#ifdef IS_DRAW_POINTS
	stringstream filename;
	filename << "out//points_" << cam_idx << ".jpg";
	imwrite(filename.str(), gray_image);
	filename.clear();
	filename.str("");	//these two statement can completely empty stringstream
	filename << "out//tag_mask_" << cam_idx << ".jpg";
	imwrite(filename.str(), tag_mask);
#endif
	cout << "> " << cam_idx << " FindPoints() done!!" << endl;
}

void TwoPass(
	const Mat &binary_image, 
	Mat &label_image
)
{
	if (binary_image.empty() ||
		binary_image.type() != CV_8UC1)
	{
		return;
	}

	label_image.release();
	binary_image.convertTo(label_image, CV_32SC1);

	int label = 1;
	vector<int> labelSet;
	labelSet.push_back(0);
	labelSet.push_back(1);

	int rows = binary_image.rows - 1;
	int cols = binary_image.cols - 1;
	for (int i = 1; i < rows; i++)	//i = 1 ~ rows-2，why?
	{
		int* data_preRow = label_image.ptr<int>(i - 1);
		int* data_curRow = label_image.ptr<int>(i);
		for (int j = 1; j < cols; j++)
		{
			if (data_curRow[j] == 1)
			{
				vector<int> neighborLabels;
				neighborLabels.reserve(2);
				int leftPixel = data_curRow[j - 1];
				int upPixel = data_preRow[j];
				if (leftPixel > 1)
				{
					neighborLabels.push_back(leftPixel);
				}
				if (upPixel > 1)
				{
					neighborLabels.push_back(upPixel);
				}

				if (neighborLabels.empty())
				{
					labelSet.push_back(++label);
					data_curRow[j] = label;
					labelSet[label] = label;
				}
				else
				{
					std::sort(neighborLabels.begin(), neighborLabels.end());
					int smallestLabel = neighborLabels[0];
					data_curRow[j] = smallestLabel;

					// 保存最小等价表
					for (size_t k = 1; k < neighborLabels.size(); k++)
					{
						int tempLabel = neighborLabels[k];
						int& oldSmallestLabel = labelSet[tempLabel];
						if (oldSmallestLabel > smallestLabel)
						{
							labelSet[oldSmallestLabel] = smallestLabel;
							oldSmallestLabel = smallestLabel;
						}
						else if (oldSmallestLabel < smallestLabel)
						{
							labelSet[smallestLabel] = oldSmallestLabel;
						}
					}
				}
			}
		}
	}

	// 更新等价对列表
	// 将最小标号给重复区域
	for (size_t i = 2; i < labelSet.size(); i++)
	{
		int curLabel = labelSet[i];
		int preLabel = labelSet[curLabel];
		while (preLabel != curLabel)
		{
			curLabel = preLabel;
			preLabel = labelSet[preLabel];
		}
		labelSet[i] = curLabel;
	}

	for (int i = 0; i < rows; i++)
	{
		int* data = label_image.ptr<int>(i);
		for (int j = 0; j < cols; j++)
		{
			int& pixelLabel = data[j];
			pixelLabel = labelSet[pixelLabel];
		}
	}
}
