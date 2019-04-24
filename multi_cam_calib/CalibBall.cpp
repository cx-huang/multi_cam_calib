#include "CalibBall.h"
#include "opencv2/cvsba.h"

#define IS_DEBUG_FINDPOINTS
//#define IS_DEBUG_CALR
//#define IS_OUTPUT_CIRCLE
//#define IS_DRAW_POINTS
#define IS_PROJ
#define MINR 1000	//in FindCircle(): the minimum radius of the circle in the image
#define MAXR 2000	//in FindCircle(): the maximum raidus of the circle in the image
#define TAG_THRES 50//in FindPoints(): for binarization of tag mask
#define SHARED_PNT_THRES 0.1	//in CalR(): for selection of shared points
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
#if 1
		for (int i = 0; i < _cam_num; i++)
		{
			int cur_npnt = cur_pntsI[i].size();
			for (int j = 0; j < cur_npnt; j++)
			{
				pntsI[i].push_back(cur_pntsI[i][j]);
				visibility[i].push_back(cur_visibility[i][j]);
			}
		}

		Mat T_base, T_new;
		Point3d T_temp;
		if (i == 0)
		{
			T_base = T[0];
			T_temp.x = 0;
			T_temp.y = 0;
			T_temp.z = 0;
		}
		else
		{
			T_new = T_base - T[0];
			T_temp.x = T_new.at<double>(0, 0);
			T_temp.y = T_new.at<double>(1, 0);
			T_temp.z = T_new.at<double>(2, 0);
		}
		for (int i = 0; i < cur_pntsW.size(); i++)
			pntsW.push_back(cur_pntsW[i] + T_temp);
#endif
	}
#if 1
	cvsba::Sba sba;
	cvsba::Sba::Params params;
	params.type = cvsba::Sba::MOTIONSTRUCTURE;
	params.iterations = 1000;
	params.minError = 1e-1;
	params.fixedIntrinsics = 1;
	params.fixedDistortion = 5;
	params.verbose = 0;
	sba.setParams(params);
	sba.run(pntsW, pntsI, visibility, cam_matrices, R, T, dis_coeffs);
	std::cout << "** SBA Optimization **" << endl << "Initial error = " << sba.getInitialReprjError() << endl << "Final error = " << sba.getFinalReprjError() << endl;
#endif
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
#ifndef IS_DEBUG_FINDPOINTS
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
		CalT(T[i], circles[i], i);
		PointI2W(pntpairsI[i], pntpairsW[i], circles[i], T[i], i);
	}

	FileStorage fw1("debug1.yml", FileStorage::WRITE);
	fw1 << "circles" << circles;
	int pntpairsW_size = pntpairsW.size();
	fw1 << "pntpairsW_size" << pntpairsW_size;
	for (int i = 0; i < pntpairsW_size; i++)
	{
		stringstream ss;
		ss << "pntpairsW_" << i;
		fw1 << ss.str() << pntpairsW[i];
	}
	int pntpairsI_size = pntpairsI.size();
	fw1 << "pntpairsI_size" << pntpairsI_size;
	for (int i = 0; i < pntpairsI_size; i++)
	{
		stringstream ss;
		ss << "pntpairsI_" << i;
		fw1 << ss.str() << pntpairsI[i];
	}
	fw1 << "T" << T;
	fw1.release();
	std::cout << ">> successfully write debug1.yml!" << endl;
#else
	FileStorage fr1("debug1.yml", FileStorage::READ);
	int pntpairsW_size_;
	fr1["pntpairsW_size"] >> pntpairsW_size_;
	pntpairsW.resize(pntpairsW_size_);
	for (int i = 0; i < pntpairsW_size_; i++)
	{
		stringstream ss;
		ss << "pntpairsW_" << i;
		fr1[ss.str()] >> pntpairsW[i];
	}
	int pntpairsI_size_;
	fr1["pntpairsI_size"] >> pntpairsI_size_;
	pntpairsI.resize(pntpairsI_size_);
	for (int i = 0; i < pntpairsI_size_; i++)
	{
		stringstream ss;
		ss << "pntpairsI_" << i;
		fr1[ss.str()] >> pntpairsI[i];
	}
	fr1["T"] >> T;
	fr1.release();
	std::cout << ">> successfully read debug1.yml!" << endl;
#endif
	//step2: calculate R
	vector<vector<Mat>> pntpairsW_normed(_cam_num);
	for (int i = 0; i < _cam_num; i++)
	{
		NormalizePoints(pntpairsW[i], pntpairsW_normed[i]);
	}
	cur_visibility[0].resize(pntpairsW_normed[0].size());
	for (int i = 0; i < cur_visibility[0].size(); i++)
	{
		cur_visibility[0][i] = -1;
	}
#ifndef IS_DEBUG_CALR
	cout << ">> calculating R ..." << endl;
	R[0] = Mat::eye(3, 3, CV_64FC1);
	for (int i = 1; i < _cam_num; i++)
	{
		CalR(pntpairsW_normed[i - 1], pntpairsW_normed[i], cur_visibility[i - 1], cur_visibility[i], cur_pntsW, R[i]);
	}

	FileStorage fw2("debug2.yml", FileStorage::WRITE);
	fw2 << "pntpairsW_normed" << pntpairsW_normed;
	fw2 << "cur_visibility" << cur_visibility;
	fw2 << "cur_pntsW" << cur_pntsW;
	fw2 << "R" << R;
	fw2.release();
	std::cout << ">> successfully write debug2.yml!" << endl;
#else
	FileStorage fr2("debug2.yml", FileStorage::WRITE);
	fr2["pntpairsW_normed"] >> pntpairsW_normed;
	fr2["cur_visibility"] >> cur_visibility;
	fr2["cur_pntsW"] >> cur_pntsW;
	fr2["R"] >> R;
	fr2.release();
	std::cout << ">> successfully read debug2.yml!" << endl;
#endif
	for (int i = 0; i < _cam_num; ++i)
		ProcessForSBA(pntpairsI[i], pntpairsW_normed[i], cur_pntsI[i], cur_pntsW, cur_visibility[i]);
}

void CalibBall::CalR(
	vector<Mat> pntpairs_src,
	vector<Mat> &pntpairs_dst,
	vector<int> &visibility_src,
	vector<int> &visibility_dst,
	vector<Point3d> &pntsW,
	Mat &R
)
{
	int src_size = pntpairs_src.size();
	int dst_size = pntpairs_dst.size();
	double threshold = SHARED_PNT_THRES;
	double min_total_error = 1e6;
	int max_shared_npnts = 0;
	int best_i, best_j;

	//step1: calculate the best R
	for (int i = 0; i < src_size; i++)
	{
		for (int j = 0; j < dst_size; j++)
		{
			Mat temp = pntpairs_dst[j] * pntpairs_src[i].t();
			Mat w, u, vt;
			SVD::compute(temp, w, u, vt, SVD::FULL_UV);
			Mat R_candidate = u * vt;

			double total_error = 0;
			int shared_npnts = 0;
			for (int k = 0; k < src_size; k++)
			{
				if (k == i)
				{
					continue;
				}

				Mat pntpair_new = R_candidate * pntpairs_src[k];
				double min_error = 1e6;
				for (int l = 0; l < dst_size; l++)
				{
					if (l == j)
					{
						continue;
					}

					double error = MAX(norm(pntpair_new.col(0) - pntpairs_dst[l].col(0)), norm(pntpair_new.col(1) - pntpairs_dst[l].col(1)));
					if (error < min_error)
					{
						min_error = error;
					}
				}
				if (min_error < threshold)
				{
					total_error += min_error;
					shared_npnts++;
				}
			}
			if (max_shared_npnts < shared_npnts || (max_shared_npnts == shared_npnts && min_total_error > total_error))
			{
				max_shared_npnts = shared_npnts;
				min_total_error = total_error;
				R = R_candidate;
				best_i = i;
				best_j = j;
			}
		}
	}
	if (max_shared_npnts == 0)
	{
		std::cout << ">> no shared points!" << endl;
		exit(-1);
	}

	//step2: determine visible points between each two cameras
	visibility_dst.resize(dst_size);
	int the_xth_visible_pntpair = 0;
	for (int j = 0; j < dst_size; j++)
	{
		if (j == best_j)
		{
			if (visibility_src[best_i] == -1)
			{
				for (int k = 0; k < 2; k++)
				{
					Mat pnt = _ball_radii * pntpairs_src[best_i].col(k);
					pntsW.push_back(Point3d(pnt.at<double>(0, 0), pnt.at<double>(1, 0), pnt.at<double>(2, 0)));
				}
				visibility_src[best_i] = the_xth_visible_pntpair;
				the_xth_visible_pntpair++;
			}
			visibility_dst[j] = visibility_src[best_i];
			continue;
		}

		Mat pntpair_new = R.t() * pntpairs_dst[j];
		double min_error = 1e6;
		int fit_pntpair = 0;
		for (int i = 0; i < src_size; i++)
		{
			double error = norm(pntpair_new.col(0) - pntpairs_src[i].col(0)) + norm(pntpair_new.col(1) - pntpairs_src[i].col(1));
			if (error < min_error)
			{
				min_error = error;
				fit_pntpair = i;
			}
		}
		if (min_error < threshold)
		{
			if (visibility_src[fit_pntpair] == -1)
			{
				for (int k = 0; k < 2; k++)
				{
					Mat pnt = _ball_radii * pntpairs_src[fit_pntpair].col(k);
					pntsW.push_back(Point3d(pnt.at<double>(0, 0), pnt.at<double>(1, 0), pnt.at<double>(2, 0)));
				}
				visibility_src[fit_pntpair] = the_xth_visible_pntpair;
				the_xth_visible_pntpair++;
			}
			visibility_dst[j] = visibility_src[fit_pntpair];
		}
		else
		{
			visibility_dst[j] = -1;
		}
	}

	for (int i = 0; i < dst_size; i++)
	{
		pntpairs_dst[i] = R.t() * pntpairs_dst[i];
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

#ifdef IS_OUTPUT_CIRCLE
	stringstream filename;
	filename << "output//contour_" << cam_idx << ".jpg";
	imwrite(filename.str(), gray);
	filename.clear();
	filename.str("");

	Mat ref1 = image.clone();
	circle(ref1, bestCircleCenter, cvRound(circle_[2]), Scalar(255, 255, 255), 5);
	filename << "output//circle_" << cam_idx << ".jpg";
	imwrite(filename.str(), ref1);
	filename.clear();
	filename.str("");
#endif
	image = dst;
#ifdef IS_OUTPUT_CIRCLE
	filename << "output//ref_cut_" << cam_idx << ".jpg";
	imwrite(filename.str(), image);
#endif
	cout << ">> " << cam_idx << " FindCircle() done!!" << endl;
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
	filename << "output//points_" << cam_idx << ".jpg";
	imwrite(filename.str(), gray_image);
	filename.clear();
	filename.str("");	//these two statement can completely empty stringstream
	filename << "output//tag_mask_" << cam_idx << ".jpg";
	imwrite(filename.str(), tag_mask);
#endif
	cout << ">> " << cam_idx << " FindPoints() done!!" << endl;
}

void CalibBall::PointI2W(
	vector<vector<Point2f>> pntpairsI,
	vector<vector<Point3d>> &pntpairsW,
	Vec3d circle_,
	Mat T,
	int cam_idx
)
{
	double scale = _ball_radii / circle_[2];
	double r_square = SQUARE(_ball_radii);
	pntpairsW.resize(pntpairsI.size());
	for (int i = 0; i < pntpairsI.size(); i++)
	{
		pntpairsW[i].resize(2);
		for (int j = 0; j < 2; j++)
		{
			double u = pntpairsI[i][j].x;	//the horizontal coordinate of the left/right point on the jth tag in the ith camera
			double v = pntpairsI[i][j].y;	//the vertical coordinate of the left/right point on the jth tag in the ith camera
			pntpairsW[i][j].z = -sqrt(r_square - SQUARE(scale) * (SQUARE(u - circle_[0]) + SQUARE(v - circle_[1])));
			pntpairsW[i][j].x = (pntpairsW[i][j].z + T.at<double>(2, 0))*(u - _principal_pnt_x[_cam_list[cam_idx]]) / _focal_len[_cam_list[cam_idx]] - T.at<double>(0, 0);	//the subtraction of T translates the point from {C} to {W}
			pntpairsW[i][j].y = (pntpairsW[i][j].z + T.at<double>(2, 0))*(v - _principal_pnt_y[_cam_list[cam_idx]]) / _focal_len[_cam_list[cam_idx]] - T.at<double>(1, 0);
		}
	}
}

void CalibBall::CalT(
	Mat &T,
	Vec3d circle_,
	int cam_idx
)
{
	double scale = _ball_radii / circle_[2];
	T = Mat(3, 1, CV_64FC1);
	T.at<double>(0, 0) = scale * (circle_[0] - _principal_pnt_x[_cam_list[cam_idx]]);
	T.at<double>(1, 0) = scale * (circle_[1] - _principal_pnt_y[_cam_list[cam_idx]]);
	T.at<double>(2, 0) = scale * _focal_len[_cam_list[cam_idx]];
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

void CalibBall::NormalizePoints(
	vector<vector<Point3d>> pntpairW,
	vector<Mat> &pntpairW_normed
)
{
	int src_size = pntpairW.size();
	pntpairW_normed.resize(src_size);
	Mat temp(3, 3, CV_64FC1);
	for (int i = 0; i < src_size; i++)
	{
		Mat(pntpairW[i][0] * (1 / norm(pntpairW[i][0]))).copyTo(temp.col(0));
		Mat(pntpairW[i][1] * (1 / norm(pntpairW[i][1]))).copyTo(temp.col(1));
		Mat(temp.col(1).cross(temp.col(0)) * (1 / norm(temp.col(1).cross(temp.col(0))))).copyTo(temp.col(2));
		pntpairW_normed[i] = Mat::zeros(3, 3, CV_64FC1);
		temp.copyTo(pntpairW_normed[i]);
	}
}

void CalibBall::ProcessForSBA(
	vector<vector<Point2f>> pntpairsI,
	vector<Mat> pntpairsW_normed,
	vector<Point2d> &pntsI,
	vector<Point3d> pntsW,
	vector<int> &visibility
)
{
	size_t nPoints = pntsW.size();	//{W}球上角点的个数
	pntsI.resize(nPoints);
	vector<int> visibility_new(nPoints);
	Point2d p(0, 0);
	for (int i = 0; i < nPoints; i++)
	{
		visibility_new[i] = 0;
		pntsI[i] = p;
	}
	int count = 0;
	double threshold = SHARED_PNT_THRES;
	cout << "visibility: ";
	int vis_size = int(visibility.size());
	vector<Mat> points_normed(pntsW.size());
	for (int i = 0; i < pntsW.size(); i++)
	{
		points_normed[i] = Mat(pntsW[i] * (1 / _ball_radii));
	}
	for (int i = 0; i < vis_size; i++)
	{
		if (visibility[i] != -1)
		{
			int idx = visibility[i] << 1;
			pntsI[idx] = pntpairsI[i][0];
			pntsI[idx + 1] = pntpairsI[i][1];
			visibility_new[idx] = 1;
			visibility_new[idx + 1] = 1;
			count++;
		}
		else
		{
			double min_error = 1e6;
			int idx = 0;
			for (int j = 0; j < nPoints; j += 2)
			{
				if (visibility_new[j] == 0)
				{
					double error = MAX(norm(pntpairsW_normed[i].col(0) - points_normed[j]), norm(pntpairsW_normed[i].col(1) - points_normed[j + 1]));
					if (min_error > error)
					{
						min_error = error;
						idx = j;
					}
				}
			}
			if (min_error < threshold)
			{
				pntsI[idx] = pntpairsI[i][0];
				pntsI[idx + 1] = pntpairsI[i][1];
				visibility_new[idx] = 1;
				visibility_new[idx + 1] = 1;
				count++;
			}
		}
	}
	cout << count << "/" << visibility.size() << endl;
	visibility = visibility_new;
}
