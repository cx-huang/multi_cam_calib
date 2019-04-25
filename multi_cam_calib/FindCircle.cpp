#include "CalibBall.h"

/*
作者：饶咖
3.0 版本 改进：
先进行图像处理：锐化（突出边缘）。
在缩略窗口分成两次点击：减小对拍摄的要求（之前球需在照片中心）。
*/
//缩略图的缩放比例,注意：尽可能倒数为有限小数
#define SCALE 0.1
//可显示的最大宽度、高度
#define SCREEN_WIDTH 1400
#define SCREEN_HIGH 700

#define CIRCLE_CUT 120

//源图像，缩略图，选中的第一个和第二个区域
Mat src, shrink_src, area, next_area;
//两个区域的对角点
Point area_p1, area_p2, next_area_p1, next_area_p2;
//选中的圆上的点
Point circle_p;
//全局变量
int idx;
//标志
bool clicked;//flag：选定圆上第二个点
int count_click;//flag：缩略窗口点击次数
//结果
Point center;
double radius;
vector<Vec3d> result;
//第二个区域窗口 鼠标事件处理函数
void next_area_mouse(int event, int x, int y, int flags, void *ustc)
{
	Point loc;
	Mat temp;//动态效果
	Point temp_p;
	Vec3d temp_result;
	//鼠标滑动的时候画圆（选定第二个边缘点后不再画）
	if (event == CV_EVENT_MOUSEMOVE && !clicked)
	{
		next_area.copyTo(temp);
		//得到当前鼠标坐标在源图像上的映射点
		loc.x = next_area_p2.x + y;
		loc.y = next_area_p2.y + x;
		//计算对于源图像的圆心、半径
		center.x = (circle_p.x + loc.x) / 2;
		center.y = (circle_p.y + loc.y) / 2;
		radius = sqrt((circle_p.x - loc.x)*(circle_p.x - loc.x) + (circle_p.y - loc.y)*(circle_p.y - loc.y)) / 2;
		//计算对于第二个区域画圆的圆心、半径
		temp_p.x = center.y - next_area_p2.y;
		temp_p.y = center.x - next_area_p2.x;
		circle(temp, temp_p, radius, Scalar(0, 0, 255));
		imshow("next_area", temp);
	}

	//左键按下
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		destroyWindow("next_area");
		//将结果保存到数组中
		temp_result[0] = center.x;
		temp_result[1] = center.y;
		temp_result[2] = radius;
		result.push_back(temp_result);
		//画圆并输出结果图像
		clicked = true;
		circle(src, center, radius, Scalar(0, 0, 255));

		stringstream storagefile;
		storagefile << "output//points//circle_" << idx << ".jpg";		
		imwrite(storagefile.str(), src);
	}
}

//第一个区域窗口 鼠标事件处理函数
void area_mouse(int event, int x, int y, int flags, void *ustc)
{
	//左键按下
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		destroyWindow("area");
		//计算得到选中的第一个圆上的点在源图的坐标
		circle_p.x = y + area_p1.x;
		circle_p.y = x + area_p1.y;

		//截取源图像上的第二个区域
		next_area = src(Rect(next_area_p1, next_area_p2));
		transpose(next_area, next_area);//旋转90°以适应电脑屏幕的显示（转置）
		imshow("next_area", next_area);

		//注册中心对称区域窗口鼠标事件回调函数
		setMouseCallback("next_area", next_area_mouse);
	}
}

//缩略图窗口 鼠标事件处理函数
void shrink_mouse(int event, int x, int y, int flags, void *ustc)
{
	Mat temp;
	//鼠标滑动的时候画最大可显示区域
	if (event == CV_EVENT_MOUSEMOVE)
	{
		//选择第一个区域
		if (count_click == 0)
		{
			shrink_src.copyTo(temp);
			area_p1.x = x;
			area_p1.y = y;
			area_p2.x = x + cvRound(SCREEN_HIGH * SCALE);
			area_p2.y = y + cvRound(SCREEN_WIDTH * SCALE);
			rectangle(temp, area_p1, area_p2, Scalar(255, 0, 0));
			imshow("shrink_src", temp);
		}
		//选择第二个区域
		else if (count_click == 1)
		{
			shrink_src.copyTo(temp);
			//先画已经选定的区域
			rectangle(temp, area_p1, area_p2, Scalar(255, 0, 0));
			//画滑动区域
			next_area_p1.x = x + SCREEN_HIGH * SCALE;
			next_area_p1.y = y + SCREEN_WIDTH * SCALE;
			next_area_p2.x = x;
			next_area_p2.y = y;
			rectangle(temp, next_area_p1, next_area_p2, Scalar(255, 0, 0));
			imshow("shrink_src", temp);
		}
	}
	//左键按下
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		if (count_click == 0)
		{
			++count_click;
		}
		else if (count_click == 1)
		{
			//尺度换算
			area_p1.x /= SCALE;
			area_p1.y /= SCALE;
			area_p2.x /= SCALE;
			area_p2.y /= SCALE;
			next_area_p1.x /= SCALE;
			next_area_p1.y /= SCALE;
			next_area_p2.x /= SCALE;
			next_area_p2.y /= SCALE;
			//关闭缩略图窗口
			destroyWindow("shrink_src");
			//截取源图像对应区域
			area = src(Rect(area_p1, area_p2));
			transpose(area, area);//旋转90°以适应电脑屏幕的显示（转置）
			imshow("area", area);
			//注册第一个选中区域窗口鼠标事件回调函数
			setMouseCallback("area", area_mouse);
		}
	}
}

void FindCircleManually(string sample_path, vector<string> imagelist, vector<Vec3d> &circles,  vector<int> cam_list)
{
	int num = imagelist.size();

	//用于锐化的拉普拉斯算子
	Mat kernel(3, 3, CV_32F, Scalar(0));
	kernel.at<float>(1, 1) = 5.0;
	kernel.at<float>(0, 1) = -1.0;
	kernel.at<float>(1, 0) = -1.0;
	kernel.at<float>(1, 2) = -1.0;
	kernel.at<float>(2, 1) = -1.0;

	for (idx = 0; idx < num; ++idx)
	{
		//读取要找圆的源图片
		stringstream filename;
		filename << sample_path << imagelist[cam_list[idx]];
		src = imread(filename.str());
		//与拉普拉斯算子进行卷积（锐化）
		filter2D(src, src, -1, kernel);//进行多次锐化可增强效果
		//缩略图
		shrink_src;
		resize(src, shrink_src, Size(src.cols*SCALE, src.rows*SCALE), 0, 0);
		//画九宫格
		line(shrink_src, Point(cvRound(shrink_src.cols / 3), 0), Point(cvRound(shrink_src.cols / 3), shrink_src.rows), Scalar(255, 255, 255));//竖1
		line(shrink_src, Point(cvRound((shrink_src.cols * 2) / 3), 0), Point(cvRound((shrink_src.cols * 2) / 3), shrink_src.rows), Scalar(255, 255, 255));//竖2
		line(shrink_src, Point(0, cvRound(shrink_src.rows / 3)), Point(shrink_src.cols, cvRound(shrink_src.rows / 3)), Scalar(255, 255, 255));//横1
		line(shrink_src, Point(0, cvRound((shrink_src.rows * 2) / 3)), Point(shrink_src.cols, cvRound((shrink_src.rows * 2) / 3)), Scalar(255, 255, 255));//横
		imshow("shrink_src", shrink_src);
		//注册缩略图窗口鼠标事件回调函数
		setMouseCallback("shrink_src", shrink_mouse);
		//标志初始化
		clicked = false;
		count_click = 0;
		waitKey(0);
	}
	circles = result;
	waitKey(0);
}

void ExtractCircle(Mat &image, Vec3d circle_)
{
	Point2i bestCircleCenter(cvRound(circle_[0]), cvRound(circle_[1]));
	Mat mask(image.size(), CV_8UC1, Scalar(0));
	circle(mask, bestCircleCenter, cvRound(circle_[2]) - CIRCLE_CUT, Scalar(255), CV_FILLED);
	Mat dst;
	image.copyTo(dst, mask);
	image = dst;
}

void getCircle(Point2d& p1, Point2d& p2, Point2d& p3, Point2d& center, double& radius)
{
	double x1 = p1.x;
	double x2 = p2.x;
	double x3 = p3.x;

	double y1 = p1.y;
	double y2 = p2.y;
	double y3 = p3.y;

	// PLEASE CHECK FOR TYPOS IN THE FORMULA :)
	center.x = (x1*x1 + y1 * y1)*(y2 - y3) + (x2*x2 + y2 * y2)*(y3 - y1) + (x3*x3 + y3 * y3)*(y1 - y2);
	center.x /= (2 * (x1*(y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2));

	center.y = (x1*x1 + y1 * y1)*(x3 - x2) + (x2*x2 + y2 * y2)*(x1 - x3) + (x3*x3 + y3 * y3)*(x2 - x1);
	center.y /= (2 * (x1*(y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2));

	radius = sqrt((center.x - x1)*(center.x - x1) + (center.y - y1)*(center.y - y1));
}

vector<Point2d> getPointPositions(Mat binaryImage)
{
	std::vector<Point2d> pointPositions;

	for (int y = 0; y < binaryImage.rows; ++y)
	{
		uchar* rowPtr = binaryImage.ptr<uchar>(y);
		for (int x = 0; x < binaryImage.cols; ++x)
		{
			if (rowPtr[x] > 0) pointPositions.push_back(Point2d(x, y));
		}
	}

	return pointPositions;
}

double evaluateCircle(Mat dt, Point2d center, double radius)
{

	double completeDistance = 0.0f;
	int counter = 0;

	double maxDist = 10.0f;   //TODO: this might depend on the size of the circle!

	double minStep = 0.001f;
	// choose samples along the circle and count inlier percentage

	//HERE IS THE TRICK that no minimum/maximum circle is used, the number of generated points along the circle depends on the radius.
	// if this is too slow for you (e.g. too many points created for each circle), increase the step parameter, but only by factor so that it still depends on the radius

	// the parameter step depends on the circle size, otherwise small circles will create more inlier on the circle
	double step = 2 * 3.14159265359f / (6.0f * radius);
	if (step < minStep) step = minStep; // TODO: find a good value here.

	//for(double t =0; t<2*3.14159265359f; t+= 0.05f) // this one which doesnt depend on the radius, is much worse!
	for (double t = 0; t < 2 * 3.14159265359f; t += step)
	{
		int cX = int(radius*cos(t) + center.x + 0.5);
		int cY = int(radius*sin(t) + center.y + 0.5);

		if (cX < dt.cols)
			if (cX >= 0)
				if (cY < dt.rows)
					if (cY >= 0)
					{
						float temp = dt.at<float>(cY, cX);
						if (temp <= maxDist)
						{
							completeDistance += temp;
							counter++;
						}
					}
	}

	return counter;
}

void DetectCircle(Mat image, Vec3d &circle_, double minCircleRadius, double maxCircleRadius)
{
	std::vector<Point2d> edgePositions;
	edgePositions = getPointPositions(image);

	// create distance transform to efficiently evaluate distance to nearest edge
	Mat dt;
	distanceTransform(255 - image, dt, CV_DIST_L1, 3);

	Point2d bestCircleCenter;
	double bestCircleRadius;
	//double bestCVal = FLT_MAX;
	double bestCVal = -1;

	int iter = 100;
	//TODO: implement some more intelligent ransac without fixed number of iterations
	int edge_size = int(edgePositions.size());
	int width = image.cols;
	int height = image.rows;
	while (1)
	{
		//RANSAC: randomly choose 3 point and create a circle:
		//TODO: choose randomly but more intelligent,
		//so that it is more likely to choose three points of a circle.
		//For example if there are many small circles, it is unlikely to randomly choose 3 points of the same circle.
		unsigned int idx1 = rand() % edge_size;
		unsigned int idx2 = rand() % edge_size;
		unsigned int idx3 = rand() % edge_size;

		// we need 3 different samples:
		if (idx1 == idx2) continue;
		if (idx1 == idx3) continue;
		if (idx3 == idx2) continue;

		// create circle from 3 points:
		Point2d center; double radius;
		getCircle(edgePositions[idx1], edgePositions[idx2], edgePositions[idx3], center, radius);

		if (radius < minCircleRadius)continue;
		if (radius > maxCircleRadius)continue;
		if (center.x > (width - radius) || center.x < radius || center.y >(height - radius) || center.y < radius)
			continue;
		//verify or falsify the circle by inlier counting:
		//double cPerc = verifyCircle(dt,center,radius, inlierSet);
		double cVal = evaluateCircle(dt, center, radius);

		if (cVal > bestCVal)
		{
			bestCVal = cVal;
			bestCircleRadius = radius;
			bestCircleCenter = center;
		}
		iter--;
		if (iter == 0) break;
	}
	circle_[0] = bestCircleCenter.x;
	circle_[1] = bestCircleCenter.y;
	circle_[2] = bestCircleRadius;
}
