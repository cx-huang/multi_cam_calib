#include "CCalibBall.h"

//#define IS_DEBUG
#define IS_OUTPUT //是否输出投影结果
#define IS_OUTPUT_CIRCLE
#define IS_PROJ
#define SCALE 2
#define CODE_LEN 5
#define MAX_COUNT 1000
#define PI 3.14159265354

#if 1
void StoreR(const vector<Mat> &R)
{
	int cam_used_num = R.size();
	ofstream outfile("R.txt", ios::app);	//ios::app从文末继续写入
											//ios::out覆盖写入
	for (int i = 0; i < cam_used_num; ++i)
	{
		outfile << "***** camera " << i << " *****" << endl;
		outfile << R[i];
		outfile << endl;
	}
	outfile << "***** done *****" << endl;
}

void StoreT(const vector<Mat> &T)
{
	int cam_used_num = T.size();
	ofstream outfile("T.txt", ios::app);
	for (int i = 0; i < cam_used_num; ++i)
	{
		outfile << "***** camera " << i << " *****" << endl;
		outfile << T[i];
		outfile << endl;
	}
	outfile << "***** done *****" << endl;
}
#endif


CCalibBall::CCalibBall(char config_name[])
{
	FileStorage fp(config_name, FileStorage::READ);
	if (fp.isOpened() == false)
	{
		printf("cannot open file %s\n", config_name);
		return;
	}
	fp["filepath"] >> filepath;
	fp["code"] >> code;
	code_num = (int)code.length() - 5;
	fp["auto"] >> IsAuto;

	fp["backgroundlist"] >> backgroundlist;
	fp["imagelist"] >> imagelist_vector;
	/*FileNodeIterator sn = fp["kinectlist"].begin();
	for (; sn != fp["kinectlist"].end(); sn++)
		kinect.push_back(Kinect(*sn));
	kinect_num = (int)kinect.size();
	*/
	/*
		string line;
		fp["kinect_info"]>>line;
		FileStorage fs(line, FileStorage::READ);
		if (fs.isOpened() == false)
		{
			printf("cannot open file %s\n", line.c_str());
			return ;
		}
		for (int i=0; i<kinect_num; i++)
		{
			char pp[MAX_COUNT];
			sprintf(pp, "ColorCameraExtrinsic-%d", i);
			fs[pp]>>kinect[i].ColorCameraExtrinsic;
			sprintf(pp, "ColorCameraIntrinsic-%d", i);
			fs[pp]>>kinect[i].ColorCameraIntrinsic;
			sprintf(pp, "DepthCameraIntrinsic-%d", i);
			fs[pp]>>kinect[i].DepthCameraIntrinsic;
		}
		fp["kinect_calib_name"]>>kinect_calib_name;
		fp["camera_calib_name"]>>camera_calib_name;
	*/
	fp["isrotated"] >> IsRotated;
	fp["camPair"] >> cam_list;
	cam_used_num = (int)cam_list.size();

	fp["radii"] >> radii;
	fp["DF"] >> DF;
	fp["init_cx"] >> init_cx;
	fp["init_cy"] >> init_cy;
	fp["initfc"] >> focal_length;
}

void CCalibBall::run(char config_name[])
{
	clock_t start = clock();
	/* 1.读取文件 */
	FileStorage fp(config_name, FileStorage::READ);
	if (fp.isOpened() == false)
	{
		printf("cannot open file %s\n", config_name);
		return ;
	}
	/* 2.变量声明和初始化 */
	vector< Mat > cameraMatrix(cam_used_num), distCoeffs(cam_used_num), R(cam_used_num), T(cam_used_num);
		//cameraMatrix(cam_used_num)：	内参矩阵3×3
		//distCoeffs(cam_used_num)：		畸变系数矩阵5×1 Q1
		//R(cam_used_num)：				旋转矩阵3×3（每个{C}相对于同一个{W}）
		//T(cam_used_num)：				平移矩阵3×1（每个{C}相对于同一个{W}）
	Mat T_base;	//表示什么？ Q3 T_base就是第1个imagelist下，{W}相对于参考相机所在{C}的平移矩阵（每个imagelist的世界坐标系原点都在变，但XYZ轴方向始终跟参考相机{C}的XYZ轴相同，而参考相机是不动的，所以{W}的XYZ轴保持不变）
	//内参矩阵、畸变系数矩阵的初始化
	for (int i=0; i<cam_used_num; i++)
	{
		cameraMatrix[i] = Mat::zeros(3,3,CV_64FC1);
		cameraMatrix[i].at<double>(0,2) = init_cx;
		cameraMatrix[i].at<double>(1,2) = init_cy;
		cameraMatrix[i].at<double>(0,0) = focal_length[cam_list[i]];
		cameraMatrix[i].at<double>(1,1) = focal_length[cam_list[i]];
		cameraMatrix[i].at<double>(2,2) = 1;
		distCoeffs[i] = Mat::zeros(5,1,CV_64FC1);	//zeros()说明假设没有畸变
	}
	NPoints = 0;	//这个成员变量用处不大
	vector<Point3d> points;								//角点的{w}坐标（错了），不知道是球上所有角点，还是仅仅参考相机的角点 Q4 points记录的是球上几乎所有角点在参考相机所在{C}下的坐标
	vector<vector<Point2d>> imagePoints(cam_used_num);	//每个相机/每张图片的角点的{I}坐标。imagePoints只能存储某个时刻的角点坐标，而实际标定时会多次拍摄，那么imagePoints存储的是哪个时刻的呢？Q2 存储的是所有时刻的，那么它存储的是所有时刻参考相机上的角点的坐标还是所有时刻所有相机上角点的坐标？ Q27 存储的是所有时刻所有相机上的角点，这些角点是经过ProcessForSba()处理过的
	vector<vector<int>> visibility(cam_used_num);		//每个相机/每张图片的角点的可见性标记（注意不是存储可以看见的角点的数量）
	vector<string> current_imagelist(cam_used_num);
	/* 3.标定 */
	//每个时刻拍摄标定球得到一个imagelist，对每个时刻的Imagelist进行处理
	for (int iv=0; iv<imagelist_vector.size(); iv++)
	{
		current_imagelist.clear();	
		fp[imagelist_vector[iv]]>>current_imagelist;
		vector<Point3d> current_points;		//这三个变量与上面三个变量是对应的
		vector<vector<Point2d>> current_imagePoints(cam_used_num);
		vector<vector<int>> current_visibility(cam_used_num);
		/* 1.对每个时刻拍摄得到的图片集进行处理 */
		NPoints += run_once(current_imagelist, current_points, current_imagePoints, current_visibility, R, T, cameraMatrix);
			//current_points记录的是球上几乎所有角点在参考相机所在{C}下的坐标，是{C}坐标
			//注意这里传入的是R和T，说明R和T并不是只通过某个时刻拍摄得到的图片集求出来的，而是综合了所有图片集求出来的，是吗？ Q5 不是，Cal3D()和RANSAC()说明了每次循环会直接覆盖之前的结果，也就是说R和T是根据最后一个imagelist求出来的
			//NPoints记录的是所有时刻所有相机拍得的图片里的点对应的三维点的总和，搜索了一下，整份工程里并没有对NPoints变量再做处理，好像没什么用
		/* 2.记录角点的{C}坐标和{I}坐标 */
		for (int i=0; i<cam_used_num; i++)
		{
			size_t current_npoints = current_points.size();
			for (int j=0; j<current_npoints; j++)
			{
				imagePoints[i].push_back(current_imagePoints[i][j]);	//存储当前时刻第i个相机上所有的角点坐标，也就是说imagePoints[i]存储的是：第i个相机第1个时刻看到的角点的坐标、第2个时刻看到的角点的坐标，……，第n个时刻看到的角点的坐标
				visibility[i].push_back(current_visibility[i][j]);
			}
		}
		/* 3.对2中得到的{C}坐标和{I}坐标进行补充 */
		Point3d T_temp;
		if (iv == 0)	//第一个imagelist时
		{
			T_base = T[0].clone();	//T[0]=[Xcc, Ycc, Zcc]T（Xcc,Ycc，Zcc是球心在参考相机所在{C}下的坐标
									//T_base就是第1个imagelist下，{W}相对于参考相机所在{C}的平移矩阵（每个imagelist的世界坐标系原点都在变，但XYZ轴方向始终跟参考相机{C}的XYZ轴相同，而参考相机是不动的，所以{W}的XYZ轴保持不变）
			T_temp.x = 0;
			T_temp.y = 0;
			T_temp.z = 0;
		}
		else
		{
			Mat T_new = T_base-T[0];	//第1个imagelist求得的T[0] - 第iv个imagelist求得的T[0]，表示第1个imagelist时求得的角点的{W}坐标所在{C}相对于第iv个imagelist时参考相机所在{C}平移了多少（注意，实际中参考相机并没有动，动的是球，球心视为世界坐标系原点
			T_temp.x = T_new.at<double>(0,0);
			T_temp.y = T_new.at<double>(1,0);
			T_temp.z = T_new.at<double>(2,0);
		}
		for (int i=0; i<current_points.size(); i++)
			points.push_back(current_points[i]+T_temp);
				//我们知道current_points记录的是当前时刻球上几乎所有角点在参考相机所在{C}下的坐标，是{C}坐标，
				//这里对于第1个imagelist，T_temp是零向量，也就是说将第一个imagelist得到的角点的{C}坐标
				//不做平移就加入到points中，所以points存储的是{C}坐标
				//把由第iv个imagelist的到的球的角点平移到第1个imagelist时参考相机所在的{C}
		//存储每个imagelist计算得到的R和T,理论上每个imagelist计算得到的R都是相同的，而T不同
		//StoreR(R);
		//StoreT(T);
	}
	/* 4.非线性优化 */ 
	//对所有imagelist标定得到的结果进行非线性优化
	cvsba::Sba sba;
	cvsba::Sba::Params params ;
	params.type = cvsba::Sba::MOTIONSTRUCTURE;	// type of sba: motionstructure(3dpoints+extrinsics), just motion(extrinsics) or just structure(3dpoints)
	params.iterations = 200;
	params.minError = 1e-10;
	//params.fixedIntrinsics = 1;
	params.fixedIntrinsics = 1;	//number of intrinsics parameters that keep fixed [0-5] (fx cx cy fy/fx s)
	params.fixedDistortion = 5;	// number of distortion parameters that keep fixed [0-5] (k1 k2 p1 p2 k3)
	params.verbose = 0;
	sba.setParams(params);
	sba.run(points, imagePoints, visibility, cameraMatrix, R, T, distCoeffs);
	cout << "Optimization. Initial error=" << sba.getInitialReprjError();
#ifdef IS_SBA_TWICE
	double error = 0;
	size_t nPoints = points.size();
	for (int i = 0; i < cam_used_num; i++)
	{
		Mat P1 = cameraMatrix[i] * R[i];
		Mat P2 = cameraMatrix[i] * T[i];
		for (int j = 0; j < nPoints; j++)
		{
			if (visibility[i][j] == 1)
			{
				Mat u = Mat(points[j]);
				u = P1 * u + P2;
				Point2d center(u.at<double>(0, 0) / u.at<double>(2, 0), u.at<double>(1, 0) / u.at<double>(2, 0));
				if (norm(center - imagePoints[i][j]) > 2)
					visibility[i][j] = 0;
			}
		}
	}

	sba.run(points, imagePoints, visibility, cameraMatrix, R, T, distCoeffs);
#endif
	//存储BA优化后得到的R和T
	StoreR(R);
	StoreT(T);
	cout<<" and Final error="<<sba.getFinalReprjError()<<std::endl;
	/* 5.重投影和尺度缩放 */
	ProjectToImg(current_imagelist, cameraMatrix, R, T, points, imagePoints, visibility);
	ScaleToWorld(points,T);
	OutputParam(cameraMatrix, R, T, distCoeffs);
	printf("time: %f min\n", double(clock()-start)/CLOCKS_PER_SEC/60);
}

int CCalibBall::run_once(const vector<string> &imagelist, vector<Point3d> &points, vector<vector<Point2d>> &imagePoints, vector<vector<int>> &visibility, vector<Mat> &R, vector<Mat> &T, vector<Mat> cameraMatrix)
{
	vector< vector< vector<Point2f> > > pointsImg(cam_used_num);
		//pointsImg：记录当前时刻，每一张照片上角点的{I}坐标
		//pointsImg.size		= 相机个数
		//pointsImg[i].size		= 第i张图片上标记面片个数
		//pointsImg[i][j].size	= 第i张图片上第j个标记面片上的点的个数，因为可能程序运行后找到的角点不只两个
		//相比较run()中的vector<vector<Point2d>> imagePoints(cam_used_num);可以看到pointsImg存储角点时以点集存储（每个标记面片上的角点构成一个点集），而imagePoints以单点存储
	vector< vector< vector<Point3d> > > points3D(cam_used_num);		
		//points3D：记录当前时刻，每一张照片上角点在各自{C}下的{C}坐标
	vector< vector <Point2f> > MarkerPosition(cam_used_num);	//干嘛用 Q6
	int *first_label = new int [cam_used_num];	//干嘛用 Q6
	vector< Vec3d > circles(cam_used_num);
												/* 1.找角点，包含了平移矩阵的求解 */
#ifndef IS_DEBUG
	FindPoints(imagelist, T, pointsImg, points3D, MarkerPosition, first_label, circles);

	FileStorage fw("debug1.yml", FileStorage::WRITE);
	fw << "circles" << circles;
	int points3D_size = points3D.size();
	fw << "points3D_size" << points3D_size;
	for (int i = 0; i < points3D_size; i++)
	{
		stringstream ss;
		ss << "points3D_" << i;
		fw << ss.str() << points3D[i];
	}
	int pointsImg_size = pointsImg.size();
	fw << "pointsImg_size" << pointsImg_size;
	for (int i = 0; i < pointsImg_size; i++)
	{
		stringstream ss;
		ss << "pointsImg_" << i;
		fw << ss.str() << pointsImg[i];
	}
	fw << "T" << T;
	fw.release();
	std::cout << "successfully write debug1.yml!" << endl;
#endif
#ifdef IS_DEBUG
	FileStorage fr("debug1.yml", FileStorage::READ);
	int points3D_size_;
	fr["points3D_size"] >> points3D_size_;
	points3D.resize(points3D_size_);
	for (int i = 0; i < points3D_size_; i++)
	{
		stringstream ss;
		ss << "points3D_" << i;
		fr[ss.str()] >> points3D[i];
	}
	int pointsImg_size_;
	fr["pointsImg_size"] >> pointsImg_size_;
	pointsImg.resize(pointsImg_size_);
	for (int i = 0; i < pointsImg_size_; i++)
	{
		stringstream ss;
		ss << "pointsImg_" << i;
		fr[ss.str()] >> pointsImg[i];
	}
	fr["T"] >> T;
	fr.release();
	std::cout << "successfully read debug1.yml!" << endl;
#endif
	/* 2.求旋转矩阵 */
	R[0] = Mat::eye(3, 3, CV_64FC1);	//假设第一个相机的旋转矩阵是单位阵，即没有旋转
	vector<vector< Matx33d >> points3D_normed(cam_used_num);	//Matx33d：3×3的double型矩阵，由points3D中每个标记面片上的两个{C}坐标（三维向量）得到一个3×3矩阵，这个矩阵的每个分量都是一个{C}下的基向量，类似于坐标系
	Points3DToMat(points3D[0], points3D_normed[0]);
	visibility[0].resize(points3D_normed[0].size());
		//visibility[0]表示第一个相机的可见性，这句话的意思是visibility[0]的大小设置为第一张图片里的标记面片的数量大小
		//visibility.size		表示	相机个数
		//visibility[i].size	表示第i个相机第j个标记面片的个数
		//visibility[i][j]		表示第i个相机的第j个标记面片的可见性
		//						比如某个标记面片P既是第i个相机的第j个面片也是第i+1个相机的第k个面片，即两个相机都可以看到这个面片，那么visibility[i][j]=visibility[i+1][k]=index，index表示这是第index个可以共同看到的面片
	for (int i=0; i<visibility[0].size(); i++)	//对第一个相机的可见性进行初始化，-1表示看不见
		visibility[0][i] = -1;
	for (int i=1; i<cam_used_num; i++)
	{
		Points3DToMat(points3D[i], points3D_normed[i]);
		RANSAC(points3D_normed[i-1], points3D_normed[i], R[i], points, visibility[i-1], visibility[i]);
			//截止至目前points（对应run()中的current_points）记录的是球上几乎所有角点（因为有些角点没有检测到）在参考相机所在{C}下的坐标，是{C}坐标
			//注意！RANSAC()会对points3D_normed[i]修改（即points3D_normed[i] = R[i]的转置 * points3D_normed[i]）
			//所以求出的R[i]是参考相机到其他每一个相机的R，而参考相机相对于{W}没有旋转，所以求出来的R[i]就是{W}相对于每个相机的R
	}
	/* 3.为sba做前期数据处理 */
	for (int i=0; i<cam_used_num; i++)
		ProcessForSba(pointsImg[i], imagePoints[i], points, points3D_normed[i], visibility[i]);
	ProjectToImg(imagelist, cameraMatrix, R, T, points, imagePoints, visibility);
	delete [] first_label;
	return int(points.size());
}

void CCalibBall::FindPoints(const vector<string> &imagelist, vector< Mat > &spheres, vector< vector< vector<Point2f> > > &pointsImg, vector< vector< vector<Point3d> > > &points3D, vector< vector <Point2f> > &MarkerPosition, int *first_label, vector< Vec3d > &circles)
{
	
#pragma omp parallel for	//OpenMP中的一个指令，表示接下来的for循环将被多线程执行，另外每次循环之间不能有关系
	for (int i=0; i<cam_used_num; i++)
	{
		/* 1.读取某台相机的前景和背景图片 */
		Mat origin = imread(filepath+backgroundlist[cam_list[i]]);
		if (origin.rows == 0 || origin.cols == 0)
		{
			printf("%s read error\n", (filepath+backgroundlist[cam_list[i]]).c_str());
			exit(-1);
		}
		Mat ref = imread(filepath+imagelist[cam_list[i]]);
		if (IsRotated[cam_list[i]]=='1')	//拍摄时有旋转，应该不是必须的，是为了适应相机的部署才需要写这部分代码（可能有的相机在搭建的装置上安装时不得不和其他相机上下颠倒）
		{
			flip(origin, origin, -1);	//左右、上下镜像
			flip(ref, ref, -1);
		}

		/* 2.求该相机/图片里的球心坐标和半径 */
		Mat BigMarker, mask;
		FindBigCircle(origin, ref, BigMarker, mask, circles[i], i);
		//KinectMarker(BigMarker, mask, MarkerPosition[i], first_label[i], i);

		/* 3.找角点 */
		FindCorners(ref, pointsImg[i], circles[i], i);
			//ref为经过裁剪后剩下一个球的图片
			//pointsImg[i]表示第i个相机/第i张图片里标记面片上的角点的坐标，类型是 vector< vector<Point2f> >
			//i表示相机序号

		/* 4.由角点的{I}坐标求对应的{C}坐标，以及相机的平移矩阵 */
		Cal3D(circles[i], spheres[i], pointsImg[i], points3D[i], focal_length[cam_list[i]]);
			//circles[i]表示第i个相机/第i张图片的圆的球心坐标及半径
			//spheres[i]表示第i个相机的平移矩阵，
			//pointsImg[i]表示第i个相机/第i张图片里标记面片上的角点的坐标
			//point3D[i]表示第i个相机/第i张图片里标记面片上的角点对应的{C}下的坐标
			//focal_length[cam_lish[i]]表示第i个相机的焦距
	}
	
}

void CCalibBall::FindBigCircle(Mat background, Mat &ref, Mat &BigMarker, Mat &mask_dst, Vec3d &circle_, int idx)
{
	int minR=1000;//700
	int maxR=2000;//1200
	Mat gray;
	absdiff(background,ref,gray);	//跟gray = background - ref得到的结果不同，暂时不知道哪里不同
	cvtColor(gray, gray, CV_BGR2GRAY);
	//	medianBlur(gray,gray,3);
	GaussianBlur( gray, gray, Size(25, 25), 2, 2);
	gray = gray > 10;
	Mat element = getStructuringElement( MORPH_ELLIPSE, Size(20, 20));
	morphologyEx(gray, gray, MORPH_CLOSE, element, Point(-1,-1), 2);
	vector<vector<Point> > contours;
	findContours(gray, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0,0));
	double max_area = 0;
	int max_ID = -1;
	for (int i=0; i<contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if (area>max_area)
		{
			max_area = area;
			max_ID = i;
		}
	}
	gray.setTo(0);
	drawContours( gray, contours, max_ID, Scalar(255) );
	DetectCircle(gray, circle_, minR, maxR);
	if(cvRound(circle_[2]) <= 0){
		printf("circle not detected in %d\n", idx);
		exit(-1);
	}
	Point2i bestCircleCenter(cvRound(circle_[0]), cvRound(circle_[1]));	
	Mat mask(ref.size(), CV_8UC1, Scalar(0));
	circle(mask, bestCircleCenter, cvRound(circle_[2])-120, Scalar(255), CV_FILLED);
	Mat dst;
	ref.copyTo(dst, mask);
	mask.setTo(0);
	drawContours( mask, contours, max_ID, Scalar(255), CV_FILLED );
	circle(mask, bestCircleCenter, cvRound(circle_[2]), Scalar(0), CV_FILLED);
	element = getStructuringElement( MORPH_ELLIPSE, Size(10, 10));
	erode(mask, mask, element);
	findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0,0));
	max_area = 0;
	max_ID = -1;
	for (int i=0; i<contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if (area>max_area)
		{
			max_area = area;
			max_ID = i;
		}
	}
	mask.setTo(0);
	drawContours( mask, contours, max_ID, Scalar(255), CV_FILLED );
	ref.copyTo(BigMarker, mask);

	cvtColor(BigMarker, BigMarker, CV_BGR2GRAY);
	mask_dst = mask;
#ifdef IS_OUTPUT_CIRCLE
	char filename[512];
	sprintf(filename, "out//temp_%d.jpg", idx);
	imwrite(filename, gray);
	sprintf(filename, "out//out_%d.jpg", idx);
	Mat ref1 = ref.clone();
	circle(ref1, bestCircleCenter, cvRound(circle_[2]), Scalar(255,255,255), 5);
	imwrite(filename, ref1);

	ref = dst;
	sprintf(filename, "out//ref%d.jpg", idx);
	imwrite(filename, ref);
#endif
	ref = dst;
}

void CCalibBall::FindCorners(Mat image, vector< vector<Point2f> > &pointsImg, Vec3d circle_, int idx)
{
	Mat element;
	Mat image_gray, image_small;
	resize(image, image_small, Size(), 1.0/SCALE, 1.0/SCALE);	//横向纵向缩小为原来的二分之一
																//为什么要resize，resize之后的像素少了，不是更不精确了吗？ Q7
	Mat image_back = image_small.clone();	//image_back没有用到，只是备份而已
	cvtColor(image_small, image_gray, CV_BGR2GRAY);
	vector<Mat> rgb;
	split(image_small, rgb);

	Mat mask = rgb[1] - rgb[2];
	mask = mask > 20;
	mask = 255 - mask;
	//imwrite("out//lable.jpg", mask);

	/* 求角点的候选点，然后筛选 */
	vector<Point2f> corners;
	goodFeaturesToTrack(image_gray, corners, 100, 0.1, 60/SCALE, mask, 5);	
		//默认情况下计算的是shi-tomasi角点
		//角点数目最大为100，角点品质因数为0.1
		//对于初选出的角点而言，如果在其周围60/2=30范围内存在其他更强角点，则将此角点删除 Q12
		//感兴趣区域是mask中的非零部分，即标记面片所在区域
		//计算协方差矩阵时的窗口大小为5，常用值为3，但是如果图像的分辨率较高则可以考虑使用较大一点的值
	Mat labelImg;
	Mat ROI = mask(Rect((int)(circle_[0] /2 - circle_[2]/2), (int)(circle_[1]/2 - circle_[2]/2), (int)(circle_[2]), (int)(circle_[2])));
	Mat mask1(ROI.size(), CV_8UC1, Scalar(1));	//mask1的大小跟mask一样，单通道（8位：0-255），每个通道的值为1
	Mat temp4label;
	mask1.copyTo(temp4label, ROI);	//mask中像素值非零的区域覆盖mask1后保存到temp4label中
									//他这么做的目的就是把mask（要么为0要么为255）中的非零区域像素幅值为1
	mask.copyTo(mask1);
	element = getStructuringElement( MORPH_ELLIPSE, Size(8, 8));
	erode(mask1, mask1, element);	//放大mask1的黑色部分（mask1要么为0，要么为255），即白色标记面片区域变小，这么做的目的是什么？ Q13

	vector<Point> corners_new;
	for (int i=0; i<corners.size(); i++)	//把标记面片区域变小后判断之前求得的角点所在位置是否为白色，如果是就是新交点，为什么要这么做？ Q14
		if (mask1.at<uchar>(int(corners[i].y), int(corners[i].x)) == 0)	//Mat.at<uchar>(row, col)，这里使用了int强制类型转换，为什么呢？我觉得没必要，角点坐标是{I}坐标，理论上是int，这里corners本身存储的点的坐标就是int
			corners_new.push_back(corners[i]);
	corners.clear();

	/* 求每个标记面片上的角点数量 */
	Two_Pass(temp4label, labelImg);		//连通域判断
										//labelImg存储的就是连通域判断后得到的图像
	vector<vector<int>> corner_label;	//记录每个标记面片包含的点的坐标
										//corner_label.size()表示标记面片的数量
										//corner_label[i].size()表示每个标记面片包含的点的个数
	map<int,int> label;
	int k=0;
	for (int i=0; i<corners_new.size(); i++)	//遍历所有的i，属于同一个连通域的角点（即temp_label相同），则属于同一标记面片
	{
		int temp_label = labelImg.at<int>(corners_new[i].y - (circle_[1] - circle_[2])/2, corners_new[i].x - (circle_[0] - circle_[2])/2);	//得到某一个角点的label值
		if (label.count(temp_label) <= 0)	//返回的是被查找元素的个数。注意：map中不存在相同元素，所以返回值只能是1或0
		{
			label[temp_label] = k++;	//如果label中没有templabel这个值(int)，那么就把templabel加入到label中，其对应的int值为k++，意思就是说，如果此前没有数到这个标记面片，就记下来这个新的面片，表示这是数到的第k++个标记面片
										//比如第一个角点的label=2，那么该角点所属的标记面片序号为k+1=0+1=1，
										//第二个角点的label=7，那么该角点所属的标记面片序号为k+1=1+1=2
										//注意temp_label并不代表标记面片的序号，label[temp_label]才是！
			vector<int> temp_vec;
			temp_vec.push_back(i);
			corner_label.push_back(temp_vec);	//记下这个标记面片，以及这个标记面片上的角点数
		}
		else //如果此前已经数过这个标记面片，那就直接将这个面片上的角点数量加一
			corner_label[label[temp_label]].push_back(i);	//label[temp_label]表示标记面片的序号
	}

	/* 舍弃角点数不是2的标记面片，区分剩下的每个标记面片内点的顺序 */
	TermCriteria criteria = TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001 );
		//配置迭代过程的终止条件
		//迭代次数40，我在网上看到的例程很多都用40
		//epsilon=0.001
	Mat I;
	cvtColor(image, I, CV_BGR2GRAY);	//注意这里是对image，即原图像进行处理！
	Point2f p(2.5, 2.5);
	float ratio = 0.1f;
	for (int i=0; i<corner_label.size(); i++)
	{
		//舍弃角点数不是2的标记面片
		if (corner_label[i].size() != 2)
			continue;
		vector<Point2f> temp(2);
		temp[0] = corners_new[corner_label[i][0]];	//某面片上的第一个角点
		temp[1] = corners_new[corner_label[i][1]];	//同一面片上的第二个角点
		cornerSubPix(image_gray, temp , Size(7,7), Size(-1,-1), criteria);
			//注意这里是对image_gray，即原图像缩小一半后的图像进行亚像素精度求解
			//搜素窗口为（7×2+1）×（7×2+1），也就是一个像素映射为49个像素吗？Q15
			//temp作为输入，也作为输出
		temp[0] = temp[0]*SCALE+p;	//因为之前是将图像缩小为一半后求得的角点坐标，而这里是对image，即原图像进行判断，所以要将坐标放大一倍，可是为什么要加p呢？p又是怎么确定的？Q16
		temp[1] = temp[1]*SCALE+p;
		cornerSubPix(I, temp , Size(5,5), Size(-1,-1), criteria);	//再次亚像素是因为之前亚像素操作处理的是缩小后的图像，求得缩小的图像的亚像素角点后，放大一倍，再进行一次亚像素
		//区分角点顺序（看不懂，觉得好厉害） Q17
		Point2f temp_point = (1-ratio)*temp[0]+ratio*temp[1];
		Point2f temp_point2 = (temp[1]-temp[0])*ratio;
		Point2f point_new1(temp_point.x-temp_point2.y, temp_point.y+temp_point2.x);	//右上角像素
		Point2f point_new2(temp_point.x+temp_point2.y, temp_point.y-temp_point2.x);	//右上角像素
		if (I.at<uchar>(point_new1) > I.at<uchar>(point_new2))
		{
			Point2f temp_swap = temp[0];
			temp[0] = temp[1];
			temp[1] = temp_swap;
		}
		pointsImg.push_back(temp);
#ifdef IS_OUTPUT
		circle(I, temp[0], 3, Scalar(255), CV_FILLED);	//I为原图像灰度化后的图像，以左边点为圆心，半径3画圆，填充白色，根据输出结果可以判断temp[0]为右点
		circle(I, temp[1], 3, Scalar(0), CV_FILLED);	
#endif
	}
#ifdef IS_OUTPUT
	char filename[512];	//奇怪，这里谁给filename赋值啊？ Q18 sprintf()啊！蠢
	sprintf(filename, "out//%d.jpg", idx);
	imwrite(filename, I);
	if (IsAuto)
	{
		sprintf(filename, "out//%d_mask.jpg", idx);
		imwrite(filename, mask1);
	}
#endif
}

void CCalibBall::Cal3D(Vec3d circles, Mat &sphere, vector< vector<Point2f> > pointsImg, vector< vector<Point3d> > &points3D, double F)
{
	double scale = radii/circles[2];
	sphere = Mat(3,1,CV_64FC1);
	sphere.at<double>(0,0) = scale*(circles[0]-init_cx);	//为什么平移矩阵是这样的？T的分量是{W}坐标，怎么可能通过两个{I}坐标乘以一个倍数得到呢？是近似吗？ Q19 在假设球心的{W}为[0,0,0]的前提下，{W}相对于每个相机的平移矩阵都是这么算的，说明文档里有说
	sphere.at<double>(1,0) = scale*(circles[1]-init_cy);
	sphere.at<double>(2,0) = scale*F;
	points3D.resize(pointsImg.size());	//pointsImg.size()表示标记面片的数量
	double r_square = square(radii);
	for (int i=0; i<pointsImg.size(); i++)
	{
		points3D[i].resize(2);
		for (int j=0; j<2; j++)
		{
			double px_img = pointsImg[i][j].x;
			double py_img =  pointsImg[i][j].y;
			points3D[i][j].z = -sqrt(r_square - square(scale) * (square(px_img - circles[0]) + square(py_img - circles[1])));
			points3D[i][j].x = (points3D[i][j].z + sphere.at<double>(2,0))*(px_img-init_cx)/F - sphere.at<double>(0,0);
			points3D[i][j].y = (points3D[i][j].z + sphere.at<double>(2,0))*(py_img-init_cy)/F - sphere.at<double>(1,0);
		}
	}
}

void CCalibBall::RANSAC(vector< Matx33d > points3D_src, vector< Matx33d > &points3D_dst, Mat &R, vector<Point3d> &point3d_final, vector<int> &visibility_src, vector<int> &visibility_dst)
{
	int src_size = int(points3D_src.size());	//第i张图片的标记面片的个数，就是3×3矩阵的个数
	int dst_size = int(points3D_dst.size());	//第i+1张图片的标记面片的个数，注意dst_size不一定等于src_size
	int max_inliers = -1, best_i, best_j;
	double threshold = 0.15;
	double min_total_error = 1e6;
	/* 求解最适合的R */
	for (int i=0; i<src_size; i++)	//四个嵌套循环，复杂度：src_size^2*dst_size^2，如果src_size跟dst_size数量级相当，则复杂度为O(n4)，所以每个视角下的标记面片的数量不能太多
	{
		for (int j=0; j<dst_size; j++)
		{
			//SVD分解求R
			Matx33d test =  points3D_dst[j] * points3D_src[i].t();	//第k+1张图片的第j个标记面片对应的矩阵 * 第k张图片的第i个标记面片对应的矩阵的转置
			Mat S,U,VT;
			SVD::compute(test,S,U,VT, SVD::FULL_UV);
			Matx33d R_test = Mat(U*VT);	//从src旋转到dst的可能旋转矩阵，即第k个相机/{C}到第k+1个相机/{C}的可能旋转矩阵
			//统计内点
			int inliers = 0;
			double total_error = 0;
			for (int l=0; l<src_size; l++)	//对src中剩余的点对进行测试
			{
				if (l==i) continue;	//因为如果是求得R的第i个点对，一定可以在下面的循环中找到一个完全重合的点对j，那么min_error为0，对total_error没有影响，所以没有必要再计算了（即内点不包括完全重合的点）
				Matx33d new_src = R_test * points3D_src[l];	//src中第l个点对旋转后得到的点对（矩阵形式）
				double min_error = 1e6;
				for (int k=0; k<dst_size; k++)	//找距离最近的点对（即min_error对应的点对）
				{
					if (k==j) continue;	//这里为什么要跳过j呢？ Q20 因为只要l不是i，那么j一定不会是l的内点（即error会很大）
					double error = MAX(norm(new_src.col(0)-points3D_dst[k].col(0)), norm(new_src.col(1)-points3D_dst[k].col(1)));	//为什么这个距离是这样求的？？
						//error是这样定义的：求两个标记面片各自的左边点的距离，右边点的距离，然后取其中的最大值
					if (min_error > error)
						min_error = error;	//记录最小误差
				}
				if (min_error < threshold)	//如果小于一定阈值，则这个点对视为内点
				{
					inliers++;
					total_error+=min_error;
				}
			}
			if ((inliers == max_inliers && min_total_error > total_error) || inliers > max_inliers )	//对于不同点对i和j求得的两个旋转矩阵，如果两种情况下统计得到的内点数相同，则取总误差小的R，如果内点数不同，取内点数多的R
			{
				min_total_error = total_error;	//记下目前最小的总误差，最多的内点数量，对应的R，以及求解出这个R的点对i和j
				max_inliers = inliers;
				R = Mat(R_test);
				best_i = i;
				best_j = j;
			}
		}
	}
	if (max_inliers == -1)	//如果一个内点都没有，退出
	{
		printf("no inliers!!!\n");
		exit(0);
	}
	/* 重新判断可见性，以增加可见点的数量 */
	Mat R_t = R.t();	//A到B的旋转矩阵=B到A的旋转矩阵的转置
	Matx33d R_temp = R_t;
	double total_error = 0;
	int points_end_idx = int(point3d_final.size())/2;
		//points_end_index表示有记录的标记面片的序号，也就是可见的标记面片的数量
		//这里int(point3d_final.size())/2是因为points3d_final.size()就是角点的数量，除以2就是标记面片的数量
		//而实际上point3d_final至此还没有赋值，所以它的size为0，points_end_index为0，我觉得直接赋值为0就好了
	visibility_dst.resize(dst_size);
	for (int k=0; k<dst_size; k++)
	{
		if (k==best_j)	//这是个特例，单独拎出来是为了减少计算，不要也可以
		{
			if (visibility_src[best_i] == -1)	//如果是best_j这个点对而且best_i这个点对没有记录，那就把best_i这个点对加入到可见点集中，给它个标号points_end_index，表示这是讨论到的第points_end_index个标记面片
			{
				for (int i=0; i<2; i++)
				{
					Mat u = radii*Mat(points3D_src[best_i].col(i));	//u记录的就是best_i这个面片对应的角点的{C}的坐标，很奇怪，points3D_src存储的变量是矩阵，矩阵的每列表示一个单位向量，这里直接用半径乘单位向量得到角点的{C}坐标，那不就隐含假设球心在{C}原点吗？ Q21
					Point3d p(u.at<double>(0,0), u.at<double>(1,0), u.at<double>(2,0));
					point3d_final.push_back(p);
				}
				visibility_src[best_i] = points_end_idx;	//把best_i这个点对加入到第k个相机的可见点集中，给它个标号points_end_index，表示这是看得见到的第points_end_index个标记面片
				points_end_idx++;
			}
			visibility_dst[best_j] = visibility_src[best_i];	//两个相机的标记面片标号相同，表示这是球上同一个面片
			continue;
		}
		//从这里开始才是重新判断可见性的主体算法
		Matx33d new_dst = R_temp * points3D_dst[k];
		double min_error = 1e6;
		int fit_l = 0;
		//求第n+1个相机中点对k对应的第n个相机中的点对l（距离小于阈值即视为对应点对）
		for (int l=0; l<src_size; l++)
		{
			//if (l==best_j) continue;	//l==best_j？？不应该是l==best_i吗？？ Q22 我觉得应该是l==best_i
			if (l == best_i) continue;
			double error = norm(new_dst.col(0)-points3D_src[l].col(0))+norm(new_dst.col(1)-points3D_src[l].col(1));
				//这个距离定义不同于之前，之前是两个距离的最大值为误差，这里是两个距离之和最为误差，为什么呢？ Q23 
				//A23 这应该是提高了准确性。之前误差取的是两个距离中的最大值，有可能两个距离很接近，这个时候只考虑了最大值，
				//只要最大值小于min_error就可以看成内点了，而这里取两个距离的和，得到的error会比只其中的最大值时大，那么就
				//就更不容易被看成内点，更不容易满足最后min_error<threshold（因为threshold前后取值相同）
			if (min_error > error)
			{
				min_error = error;
				fit_l = l;
			}
		}
		if (min_error < threshold)	//如果找到k点对对应的点对l（误差小于阈值即视为找到），那就标记一下这是球上同一个面片，表示这个面片两个视角下都可以看到
		{
			if (visibility_src[fit_l] == -1)
			{
				for (int i=0; i<2; i++)
				{
					Mat u = radii*Mat(points3D_src[fit_l].col(i));
					Point3d p(u.at<double>(0,0), u.at<double>(1,0), u.at<double>(2,0));
					point3d_final.push_back(p);
				}
				visibility_src[fit_l] = points_end_idx;
				points_end_idx++;
			}
			visibility_dst[k] = visibility_src[fit_l];
		}
		else //如果没有找到，说明第n+1个相机上的这个点对k在第n个相机中是看不见的
			visibility_dst[k] = -1;
	}
	for (int k = 0; k < dst_size; k++)
		points3D_dst[k] = R_temp * points3D_dst[k];
			//注意！这里对第n+1个相机的点对进行了修改！！！
			//首先两个相机肯定有些角点是彼此看不见的，有些是彼此都可以看见的，可以看见的才是我们要讨论的，
			//那么目的相机乘旋转矩阵的逆之后的情况就是原先彼此都可以看见的点（几乎）重合了，彼此看不见的点随它便，
			//可以看到point3d_final push的都是通过pointd3D_src[]计算得到的点，而目的相机又会作为下一次循环的源相机，
			//那么也就是说我们是在增加一个视角下相机的可见点的数量，即参考相机的视角，而不是所有视角！
			//Q24：但是这里有个问题，目的相机中那些参考相机看不见的角点，会在下一次循环中被当作参考相机上的角点来讨论，那不就错了吗？
			//我觉得不应该把目的相机乘以逆矩阵，而应该直接将目的相机赋值为源相机（可以用另外的变量备份赋值前的目的相机角点）
			//我用前四张照片，运行points3D_dst[k] = points3D_src[k];报错数组溢出，因为目的相机的标记面片数量本来就不等于源相机的标记面片的数量
			//A25 不是的！points3d_final记录的是球上大部分角点（之所以说大部分是因为可能有些角点程序没有检测到）在参考相机所在{C}的坐标
			//RANSAC()牛逼！
}
			
void CCalibBall::Two_Pass(const Mat& binImg, Mat& lableImg)    //两遍扫描法
{
	if (binImg.empty() ||
		binImg.type() != CV_8UC1)
	{
		return;
	}

	// 第一个通路

	lableImg.release();	//调用析构函数，传进来时labelImg还没有初始化，它应该是作为一个容器，一个壳
	binImg.convertTo(lableImg, CV_32SC1);

	int label = 1; 
	vector<int> labelSet;
	labelSet.push_back(0);  
	labelSet.push_back(1);  

	int rows = binImg.rows - 1;
	int cols = binImg.cols - 1;
	for (int i = 1; i < rows; i++)	//i = 1 ~ rows-2，why?
	{
		int* data_preRow = lableImg.ptr<int>(i-1);
		int* data_curRow = lableImg.ptr<int>(i);
		for (int j = 1; j < cols; j++)
		{
			if (data_curRow[j] == 1)
			{
				vector<int> neighborLabels;	//此时数组neighborLabels的size为0，capacity为0
				neighborLabels.reserve(2);	//此时数组neighborLabels的size为0，capacity为2，有必要吗？
				int leftPixel = data_curRow[j-1];
				int upPixel = data_preRow[j];
				if ( leftPixel > 1)
				{
					neighborLabels.push_back(leftPixel);
				}
				if (upPixel > 1)
				{
					neighborLabels.push_back(upPixel);
				}

				if (neighborLabels.empty())
				{
					labelSet.push_back(++label);  // 不连通，标签+1
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
		int* data = lableImg.ptr<int>(i);
		for (int j = 0; j < cols; j++)
		{
			int& pixelLabel = data[j];
			pixelLabel = labelSet[pixelLabel];	
		}
	}
}

void CCalibBall::Points3DToMat(vector< vector<Point3d> > points3D, vector<Matx33d> &points)
{
	int src_size = int(points3D.size());	//第i张图片的标记面片的个数
	points.resize(src_size);
	Mat temp(3,3,CV_64FC1);
	for (int i=0; i<src_size; i++)
	{
		Mat(points3D[i][0] * (1/norm(points3D[i][0]))).copyTo(temp.col(0));	//norm(points3D[i][0])求第一个三维点的L2范数，即欧几里得范数sqrt(x1^2+x2^2+...+xn^2)
		
																			//Mat(const cv::Point3d &ptr)将三维点的xyz坐标写成列向量
		Mat(points3D[i][1] * (1/norm(points3D[i][1]))).copyTo(temp.col(1));
		Mat(temp.col(1).cross(temp.col(0))).copyTo(temp.col(2));	//cross叉积，结果是垂直于向量a和b的c向量（即法向量）
		temp.col(2) /= norm(temp.col(2));
		temp.copyTo(points[i]);	//第i个标记面片对应的3×3矩阵
	}
}

void CCalibBall::ProcessForSba(vector<vector<Point2f>> pointImg_src, vector<Point2d> &pointImg_dst,  vector< Point3d> points, vector< Matx33d > points3D_normed, vector<int> &visibility)
{
	size_t nPoints = points.size();
	pointImg_dst.resize(nPoints);
	vector<int> visibility_new(nPoints);
	Point2d p(0,0);
	for (int i=0; i< nPoints; i++)
	{
		visibility_new[i] = 0;
		pointImg_dst[i] = p;
	}
	int count = 0;
	double threshold = 0.15;
	printf("visibility: ");
	int vis_size = int(visibility.size());
	vector<Mat> points_normed(points.size());
	for (int i=0; i<points.size(); i++)
	{
		points_normed[i] = Mat(points[i]*(1/radii));
	}
	for (int i=0; i<vis_size;  i++)
	{
		//		printf("%d ",visibility[i]);
		if (visibility[i] != -1)
		{
			int idx = visibility[i]<<1;
			pointImg_dst[idx] = pointImg_src[i][0];
			pointImg_dst[idx+1] = pointImg_src[i][1];
			visibility_new[idx] = 1;
			visibility_new[idx+1] = 1;
			count++;
		}
		else
		{
			double min_error = 1e6;
			int idx = 0;
			for (int j=0; j<nPoints; j+=2)
			{
				if (visibility_new[j]==0)
				{
					double error = MAX(norm(Mat(points3D_normed[i].col(0))-points_normed[j]), norm(Mat(points3D_normed[i].col(1))-points_normed[j+1]));
					if (min_error > error)
					{
						min_error = error;
						idx = j;
					}
				}
			}
			if (min_error < threshold)
			{
				pointImg_dst[idx] = pointImg_src[i][0];
				pointImg_dst[idx+1] = pointImg_src[i][1];
				visibility_new[idx] = 1;
				visibility_new[idx+1] = 1;
				count++;
			}
		}
	}
	printf("%d/%d\n", count, visibility.size());
	visibility = visibility_new;
}

void CCalibBall::OutputParam(vector< Mat > cameraMatrix, vector< Mat > R, vector< Mat > T, vector< Mat > distCoeffs)
{
	FileStorage fp(filepath+camera_calib_name, FileStorage::WRITE);
	Mat extrisic(3,4,CV_64FC1);
	for (int i=0; i<cam_used_num; i++)
	{
		string currentID = to_string((long long)cam_list[i]);
		fp<<"intrinsic-"+currentID<<cameraMatrix[i];
		R[i].copyTo(extrisic.colRange(0,3));
		T[i].copyTo(extrisic.col(3));
		fp<<"extrinsic-"+currentID<<extrisic;
	}
	fp.release();
}

void CCalibBall::ProjectToImg(const vector<string> &imagelist, vector< Mat > cameraMatrix, vector< Mat > R, vector< Mat > T, vector< Point3d> points, vector<vector< Point2d >> imagePoints, vector<vector<int>> visibility)
{
	double error = 0;
	double mean_radii = 0;
	int nvis = 0;
	size_t nPoints = points.size();
	for (int i=0; i<cam_used_num; i++)
	{
#ifdef IS_PROJ
		bool rotate90 = IsRotated[cam_list[i]] == '0';
		Mat ref = imread(filepath+imagelist[cam_list[i]]);
		if (IsRotated[cam_list[i]]=='1')
			flip(ref, ref, -1);
#endif
		Mat P1 = cameraMatrix[i]*R[i];
		Mat P2 = cameraMatrix[i]*T[i];
		for (int j=0; j<nPoints; j++)
		{
			if (visibility[i][j] == 1)
			{
				Mat u = Mat(points[j]);
				u = P1*u+P2;
				Point2d center(u.at<double>(0,0)/u.at<double>(2,0), u.at<double>(1,0)/u.at<double>(2,0));
#ifdef IS_PROJ
				circle(ref, Point(center), 5, Scalar(0,255,0), CV_FILLED);
				circle(ref, Point(imagePoints[i][j]), 4, Scalar(255,0,0), CV_FILLED);
#endif
				error += norm(center-imagePoints[i][j]);

				nvis ++;
			}
		}
#ifdef IS_PROJ
		imwrite("projection//"+imagelist[cam_list[i]], ref);
#endif
	}
	cout << "total error: " << error << endl;
	printf("mean square error: %f of %d projections\n", error/nvis, nvis);
	for (int i=0; i<nPoints; i++)
		mean_radii += norm(points[i]);
	printf("mean radii: %f of %d points\n", mean_radii/nPoints, nPoints);
}

void CCalibBall::ScaleToWorld(vector< Point3d> &points, vector< Mat > &T)
{
	double mean_DF = 0;
	size_t nPoints = points.size();
	for (int i=0; i<nPoints; i+=2)
	{
		mean_DF += norm(points[i+1]-points[i]);
	}
	double scale = DF/(mean_DF/nPoints*2);
	for (int i=0; i<cam_used_num; i++)
	{
		T[i] *= scale;
	}
	for (int i=0; i<nPoints; i++)
	{
		points[i] *= scale;
	}
}

CCalibBall::~CCalibBall()
{
}
