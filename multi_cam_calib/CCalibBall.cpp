#include "CCalibBall.h"

//#define IS_DEBUG
#define IS_OUTPUT //�Ƿ����ͶӰ���
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
	ofstream outfile("R.txt", ios::app);	//ios::app����ĩ����д��
											//ios::out����д��
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
	/* 1.��ȡ�ļ� */
	FileStorage fp(config_name, FileStorage::READ);
	if (fp.isOpened() == false)
	{
		printf("cannot open file %s\n", config_name);
		return ;
	}
	/* 2.���������ͳ�ʼ�� */
	vector< Mat > cameraMatrix(cam_used_num), distCoeffs(cam_used_num), R(cam_used_num), T(cam_used_num);
		//cameraMatrix(cam_used_num)��	�ڲξ���3��3
		//distCoeffs(cam_used_num)��		����ϵ������5��1 Q1
		//R(cam_used_num)��				��ת����3��3��ÿ��{C}�����ͬһ��{W}��
		//T(cam_used_num)��				ƽ�ƾ���3��1��ÿ��{C}�����ͬһ��{W}��
	Mat T_base;	//��ʾʲô�� Q3 T_base���ǵ�1��imagelist�£�{W}����ڲο��������{C}��ƽ�ƾ���ÿ��imagelist����������ϵԭ�㶼�ڱ䣬��XYZ�᷽��ʼ�ո��ο����{C}��XYZ����ͬ�����ο�����ǲ����ģ�����{W}��XYZ�ᱣ�ֲ��䣩
	//�ڲξ��󡢻���ϵ������ĳ�ʼ��
	for (int i=0; i<cam_used_num; i++)
	{
		cameraMatrix[i] = Mat::zeros(3,3,CV_64FC1);
		cameraMatrix[i].at<double>(0,2) = init_cx;
		cameraMatrix[i].at<double>(1,2) = init_cy;
		cameraMatrix[i].at<double>(0,0) = focal_length[cam_list[i]];
		cameraMatrix[i].at<double>(1,1) = focal_length[cam_list[i]];
		cameraMatrix[i].at<double>(2,2) = 1;
		distCoeffs[i] = Mat::zeros(5,1,CV_64FC1);	//zeros()˵������û�л���
	}
	NPoints = 0;	//�����Ա�����ô�����
	vector<Point3d> points;								//�ǵ��{w}���꣨���ˣ�����֪�����������нǵ㣬���ǽ����ο�����Ľǵ� Q4 points��¼�������ϼ������нǵ��ڲο��������{C}�µ�����
	vector<vector<Point2d>> imagePoints(cam_used_num);	//ÿ�����/ÿ��ͼƬ�Ľǵ��{I}���ꡣimagePointsֻ�ܴ洢ĳ��ʱ�̵Ľǵ����꣬��ʵ�ʱ궨ʱ�������㣬��ôimagePoints�洢�����ĸ�ʱ�̵��أ�Q2 �洢��������ʱ�̵ģ���ô���洢��������ʱ�̲ο�����ϵĽǵ�����껹������ʱ����������Ͻǵ�����ꣿ Q27 �洢��������ʱ����������ϵĽǵ㣬��Щ�ǵ��Ǿ���ProcessForSba()�������
	vector<vector<int>> visibility(cam_used_num);		//ÿ�����/ÿ��ͼƬ�Ľǵ�Ŀɼ��Ա�ǣ�ע�ⲻ�Ǵ洢���Կ����Ľǵ��������
	vector<string> current_imagelist(cam_used_num);
	/* 3.�궨 */
	//ÿ��ʱ������궨��õ�һ��imagelist����ÿ��ʱ�̵�Imagelist���д���
	for (int iv=0; iv<imagelist_vector.size(); iv++)
	{
		current_imagelist.clear();	
		fp[imagelist_vector[iv]]>>current_imagelist;
		vector<Point3d> current_points;		//�������������������������Ƕ�Ӧ��
		vector<vector<Point2d>> current_imagePoints(cam_used_num);
		vector<vector<int>> current_visibility(cam_used_num);
		/* 1.��ÿ��ʱ������õ���ͼƬ�����д��� */
		NPoints += run_once(current_imagelist, current_points, current_imagePoints, current_visibility, R, T, cameraMatrix);
			//current_points��¼�������ϼ������нǵ��ڲο��������{C}�µ����꣬��{C}����
			//ע�����ﴫ�����R��T��˵��R��T������ֻͨ��ĳ��ʱ������õ���ͼƬ��������ģ������ۺ�������ͼƬ��������ģ����� Q5 ���ǣ�Cal3D()��RANSAC()˵����ÿ��ѭ����ֱ�Ӹ���֮ǰ�Ľ����Ҳ����˵R��T�Ǹ������һ��imagelist�������
			//NPoints��¼��������ʱ����������ĵõ�ͼƬ��ĵ��Ӧ����ά����ܺͣ�������һ�£����ݹ����ﲢû�ж�NPoints����������������ûʲô��
		/* 2.��¼�ǵ��{C}�����{I}���� */
		for (int i=0; i<cam_used_num; i++)
		{
			size_t current_npoints = current_points.size();
			for (int j=0; j<current_npoints; j++)
			{
				imagePoints[i].push_back(current_imagePoints[i][j]);	//�洢��ǰʱ�̵�i����������еĽǵ����꣬Ҳ����˵imagePoints[i]�洢���ǣ���i�������1��ʱ�̿����Ľǵ�����ꡢ��2��ʱ�̿����Ľǵ�����꣬��������n��ʱ�̿����Ľǵ������
				visibility[i].push_back(current_visibility[i][j]);
			}
		}
		/* 3.��2�еõ���{C}�����{I}������в��� */
		Point3d T_temp;
		if (iv == 0)	//��һ��imagelistʱ
		{
			T_base = T[0].clone();	//T[0]=[Xcc, Ycc, Zcc]T��Xcc,Ycc��Zcc�������ڲο��������{C}�µ�����
									//T_base���ǵ�1��imagelist�£�{W}����ڲο��������{C}��ƽ�ƾ���ÿ��imagelist����������ϵԭ�㶼�ڱ䣬��XYZ�᷽��ʼ�ո��ο����{C}��XYZ����ͬ�����ο�����ǲ����ģ�����{W}��XYZ�ᱣ�ֲ��䣩
			T_temp.x = 0;
			T_temp.y = 0;
			T_temp.z = 0;
		}
		else
		{
			Mat T_new = T_base-T[0];	//��1��imagelist��õ�T[0] - ��iv��imagelist��õ�T[0]����ʾ��1��imagelistʱ��õĽǵ��{W}��������{C}����ڵ�iv��imagelistʱ�ο��������{C}ƽ���˶��٣�ע�⣬ʵ���вο������û�ж�����������������Ϊ��������ϵԭ��
			T_temp.x = T_new.at<double>(0,0);
			T_temp.y = T_new.at<double>(1,0);
			T_temp.z = T_new.at<double>(2,0);
		}
		for (int i=0; i<current_points.size(); i++)
			points.push_back(current_points[i]+T_temp);
				//����֪��current_points��¼���ǵ�ǰʱ�����ϼ������нǵ��ڲο��������{C}�µ����꣬��{C}���꣬
				//������ڵ�1��imagelist��T_temp����������Ҳ����˵����һ��imagelist�õ��Ľǵ��{C}����
				//����ƽ�ƾͼ��뵽points�У�����points�洢����{C}����
				//���ɵ�iv��imagelist�ĵ�����Ľǵ�ƽ�Ƶ���1��imagelistʱ�ο�������ڵ�{C}
		//�洢ÿ��imagelist����õ���R��T,������ÿ��imagelist����õ���R������ͬ�ģ���T��ͬ
		//StoreR(R);
		//StoreT(T);
	}
	/* 4.�������Ż� */ 
	//������imagelist�궨�õ��Ľ�����з������Ż�
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
	//�洢BA�Ż���õ���R��T
	StoreR(R);
	StoreT(T);
	cout<<" and Final error="<<sba.getFinalReprjError()<<std::endl;
	/* 5.��ͶӰ�ͳ߶����� */
	ProjectToImg(current_imagelist, cameraMatrix, R, T, points, imagePoints, visibility);
	ScaleToWorld(points,T);
	OutputParam(cameraMatrix, R, T, distCoeffs);
	printf("time: %f min\n", double(clock()-start)/CLOCKS_PER_SEC/60);
}

int CCalibBall::run_once(const vector<string> &imagelist, vector<Point3d> &points, vector<vector<Point2d>> &imagePoints, vector<vector<int>> &visibility, vector<Mat> &R, vector<Mat> &T, vector<Mat> cameraMatrix)
{
	vector< vector< vector<Point2f> > > pointsImg(cam_used_num);
		//pointsImg����¼��ǰʱ�̣�ÿһ����Ƭ�Ͻǵ��{I}����
		//pointsImg.size		= �������
		//pointsImg[i].size		= ��i��ͼƬ�ϱ����Ƭ����
		//pointsImg[i][j].size	= ��i��ͼƬ�ϵ�j�������Ƭ�ϵĵ�ĸ�������Ϊ���ܳ������к��ҵ��Ľǵ㲻ֻ����
		//��Ƚ�run()�е�vector<vector<Point2d>> imagePoints(cam_used_num);���Կ���pointsImg�洢�ǵ�ʱ�Ե㼯�洢��ÿ�������Ƭ�ϵĽǵ㹹��һ���㼯������imagePoints�Ե���洢
	vector< vector< vector<Point3d> > > points3D(cam_used_num);		
		//points3D����¼��ǰʱ�̣�ÿһ����Ƭ�Ͻǵ��ڸ���{C}�µ�{C}����
	vector< vector <Point2f> > MarkerPosition(cam_used_num);	//������ Q6
	int *first_label = new int [cam_used_num];	//������ Q6
	vector< Vec3d > circles(cam_used_num);
												/* 1.�ҽǵ㣬������ƽ�ƾ������� */
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
	/* 2.����ת���� */
	R[0] = Mat::eye(3, 3, CV_64FC1);	//�����һ���������ת�����ǵ�λ�󣬼�û����ת
	vector<vector< Matx33d >> points3D_normed(cam_used_num);	//Matx33d��3��3��double�;�����points3D��ÿ�������Ƭ�ϵ�����{C}���꣨��ά�������õ�һ��3��3������������ÿ����������һ��{C}�µĻ�����������������ϵ
	Points3DToMat(points3D[0], points3D_normed[0]);
	visibility[0].resize(points3D_normed[0].size());
		//visibility[0]��ʾ��һ������Ŀɼ��ԣ���仰����˼��visibility[0]�Ĵ�С����Ϊ��һ��ͼƬ��ı����Ƭ��������С
		//visibility.size		��ʾ	�������
		//visibility[i].size	��ʾ��i�������j�������Ƭ�ĸ���
		//visibility[i][j]		��ʾ��i������ĵ�j�������Ƭ�Ŀɼ���
		//						����ĳ�������ƬP���ǵ�i������ĵ�j����ƬҲ�ǵ�i+1������ĵ�k����Ƭ����������������Կ��������Ƭ����ôvisibility[i][j]=visibility[i+1][k]=index��index��ʾ���ǵ�index�����Թ�ͬ��������Ƭ
	for (int i=0; i<visibility[0].size(); i++)	//�Ե�һ������Ŀɼ��Խ��г�ʼ����-1��ʾ������
		visibility[0][i] = -1;
	for (int i=1; i<cam_used_num; i++)
	{
		Points3DToMat(points3D[i], points3D_normed[i]);
		RANSAC(points3D_normed[i-1], points3D_normed[i], R[i], points, visibility[i-1], visibility[i]);
			//��ֹ��Ŀǰpoints����Ӧrun()�е�current_points����¼�������ϼ������нǵ㣨��Ϊ��Щ�ǵ�û�м�⵽���ڲο��������{C}�µ����꣬��{C}����
			//ע�⣡RANSAC()���points3D_normed[i]�޸ģ���points3D_normed[i] = R[i]��ת�� * points3D_normed[i]��
			//���������R[i]�ǲο����������ÿһ�������R�����ο���������{W}û����ת�������������R[i]����{W}�����ÿ�������R
	}
	/* 3.Ϊsba��ǰ�����ݴ��� */
	for (int i=0; i<cam_used_num; i++)
		ProcessForSba(pointsImg[i], imagePoints[i], points, points3D_normed[i], visibility[i]);
	ProjectToImg(imagelist, cameraMatrix, R, T, points, imagePoints, visibility);
	delete [] first_label;
	return int(points.size());
}

void CCalibBall::FindPoints(const vector<string> &imagelist, vector< Mat > &spheres, vector< vector< vector<Point2f> > > &pointsImg, vector< vector< vector<Point3d> > > &points3D, vector< vector <Point2f> > &MarkerPosition, int *first_label, vector< Vec3d > &circles)
{
	
#pragma omp parallel for	//OpenMP�е�һ��ָ���ʾ��������forѭ���������߳�ִ�У�����ÿ��ѭ��֮�䲻���й�ϵ
	for (int i=0; i<cam_used_num; i++)
	{
		/* 1.��ȡĳ̨�����ǰ���ͱ���ͼƬ */
		Mat origin = imread(filepath+backgroundlist[cam_list[i]]);
		if (origin.rows == 0 || origin.cols == 0)
		{
			printf("%s read error\n", (filepath+backgroundlist[cam_list[i]]).c_str());
			exit(-1);
		}
		Mat ref = imread(filepath+imagelist[cam_list[i]]);
		if (IsRotated[cam_list[i]]=='1')	//����ʱ����ת��Ӧ�ò��Ǳ���ģ���Ϊ����Ӧ����Ĳ������Ҫд�ⲿ�ִ��루�����е�����ڴ��װ���ϰ�װʱ���ò�������������µߵ���
		{
			flip(origin, origin, -1);	//���ҡ����¾���
			flip(ref, ref, -1);
		}

		/* 2.������/ͼƬ�����������Ͱ뾶 */
		Mat BigMarker, mask;
		FindBigCircle(origin, ref, BigMarker, mask, circles[i], i);
		//KinectMarker(BigMarker, mask, MarkerPosition[i], first_label[i], i);

		/* 3.�ҽǵ� */
		FindCorners(ref, pointsImg[i], circles[i], i);
			//refΪ�����ü���ʣ��һ�����ͼƬ
			//pointsImg[i]��ʾ��i�����/��i��ͼƬ������Ƭ�ϵĽǵ�����꣬������ vector< vector<Point2f> >
			//i��ʾ������

		/* 4.�ɽǵ��{I}�������Ӧ��{C}���꣬�Լ������ƽ�ƾ��� */
		Cal3D(circles[i], spheres[i], pointsImg[i], points3D[i], focal_length[cam_list[i]]);
			//circles[i]��ʾ��i�����/��i��ͼƬ��Բ���������꼰�뾶
			//spheres[i]��ʾ��i�������ƽ�ƾ���
			//pointsImg[i]��ʾ��i�����/��i��ͼƬ������Ƭ�ϵĽǵ������
			//point3D[i]��ʾ��i�����/��i��ͼƬ������Ƭ�ϵĽǵ��Ӧ��{C}�µ�����
			//focal_length[cam_lish[i]]��ʾ��i������Ľ���
	}
	
}

void CCalibBall::FindBigCircle(Mat background, Mat &ref, Mat &BigMarker, Mat &mask_dst, Vec3d &circle_, int idx)
{
	int minR=1000;//700
	int maxR=2000;//1200
	Mat gray;
	absdiff(background,ref,gray);	//��gray = background - ref�õ��Ľ����ͬ����ʱ��֪�����ﲻͬ
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
	resize(image, image_small, Size(), 1.0/SCALE, 1.0/SCALE);	//����������СΪԭ���Ķ���֮һ
																//ΪʲôҪresize��resize֮����������ˣ����Ǹ�����ȷ���� Q7
	Mat image_back = image_small.clone();	//image_backû���õ���ֻ�Ǳ��ݶ���
	cvtColor(image_small, image_gray, CV_BGR2GRAY);
	vector<Mat> rgb;
	split(image_small, rgb);

	Mat mask = rgb[1] - rgb[2];
	mask = mask > 20;
	mask = 255 - mask;
	//imwrite("out//lable.jpg", mask);

	/* ��ǵ�ĺ�ѡ�㣬Ȼ��ɸѡ */
	vector<Point2f> corners;
	goodFeaturesToTrack(image_gray, corners, 100, 0.1, 60/SCALE, mask, 5);	
		//Ĭ������¼������shi-tomasi�ǵ�
		//�ǵ���Ŀ���Ϊ100���ǵ�Ʒ������Ϊ0.1
		//���ڳ�ѡ���Ľǵ���ԣ����������Χ60/2=30��Χ�ڴ���������ǿ�ǵ㣬�򽫴˽ǵ�ɾ�� Q12
		//����Ȥ������mask�еķ��㲿�֣��������Ƭ��������
		//����Э�������ʱ�Ĵ��ڴ�СΪ5������ֵΪ3���������ͼ��ķֱ��ʽϸ�����Կ���ʹ�ýϴ�һ���ֵ
	Mat labelImg;
	Mat ROI = mask(Rect((int)(circle_[0] /2 - circle_[2]/2), (int)(circle_[1]/2 - circle_[2]/2), (int)(circle_[2]), (int)(circle_[2])));
	Mat mask1(ROI.size(), CV_8UC1, Scalar(1));	//mask1�Ĵ�С��maskһ������ͨ����8λ��0-255����ÿ��ͨ����ֵΪ1
	Mat temp4label;
	mask1.copyTo(temp4label, ROI);	//mask������ֵ��������򸲸�mask1�󱣴浽temp4label��
									//����ô����Ŀ�ľ��ǰ�mask��ҪôΪ0ҪôΪ255���еķ����������ط�ֵΪ1
	mask.copyTo(mask1);
	element = getStructuringElement( MORPH_ELLIPSE, Size(8, 8));
	erode(mask1, mask1, element);	//�Ŵ�mask1�ĺ�ɫ���֣�mask1ҪôΪ0��ҪôΪ255��������ɫ�����Ƭ�����С����ô����Ŀ����ʲô�� Q13

	vector<Point> corners_new;
	for (int i=0; i<corners.size(); i++)	//�ѱ����Ƭ�����С���ж�֮ǰ��õĽǵ�����λ���Ƿ�Ϊ��ɫ������Ǿ����½��㣬ΪʲôҪ��ô���� Q14
		if (mask1.at<uchar>(int(corners[i].y), int(corners[i].x)) == 0)	//Mat.at<uchar>(row, col)������ʹ����intǿ������ת����Ϊʲô�أ��Ҿ���û��Ҫ���ǵ�������{I}���꣬��������int������corners����洢�ĵ���������int
			corners_new.push_back(corners[i]);
	corners.clear();

	/* ��ÿ�������Ƭ�ϵĽǵ����� */
	Two_Pass(temp4label, labelImg);		//��ͨ���ж�
										//labelImg�洢�ľ�����ͨ���жϺ�õ���ͼ��
	vector<vector<int>> corner_label;	//��¼ÿ�������Ƭ�����ĵ������
										//corner_label.size()��ʾ�����Ƭ������
										//corner_label[i].size()��ʾÿ�������Ƭ�����ĵ�ĸ���
	map<int,int> label;
	int k=0;
	for (int i=0; i<corners_new.size(); i++)	//�������е�i������ͬһ����ͨ��Ľǵ㣨��temp_label��ͬ����������ͬһ�����Ƭ
	{
		int temp_label = labelImg.at<int>(corners_new[i].y - (circle_[1] - circle_[2])/2, corners_new[i].x - (circle_[0] - circle_[2])/2);	//�õ�ĳһ���ǵ��labelֵ
		if (label.count(temp_label) <= 0)	//���ص��Ǳ�����Ԫ�صĸ�����ע�⣺map�в�������ͬԪ�أ����Է���ֵֻ����1��0
		{
			label[temp_label] = k++;	//���label��û��templabel���ֵ(int)����ô�Ͱ�templabel���뵽label�У����Ӧ��intֵΪk++����˼����˵�������ǰû��������������Ƭ���ͼ���������µ���Ƭ����ʾ���������ĵ�k++�������Ƭ
										//�����һ���ǵ��label=2����ô�ýǵ������ı����Ƭ���Ϊk+1=0+1=1��
										//�ڶ����ǵ��label=7����ô�ýǵ������ı����Ƭ���Ϊk+1=1+1=2
										//ע��temp_label������������Ƭ����ţ�label[temp_label]���ǣ�
			vector<int> temp_vec;
			temp_vec.push_back(i);
			corner_label.push_back(temp_vec);	//������������Ƭ���Լ���������Ƭ�ϵĽǵ���
		}
		else //�����ǰ�Ѿ�������������Ƭ���Ǿ�ֱ�ӽ������Ƭ�ϵĽǵ�������һ
			corner_label[label[temp_label]].push_back(i);	//label[temp_label]��ʾ�����Ƭ�����
	}

	/* �����ǵ�������2�ı����Ƭ������ʣ�µ�ÿ�������Ƭ�ڵ��˳�� */
	TermCriteria criteria = TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001 );
		//���õ������̵���ֹ����
		//��������40���������Ͽ��������̺ܶ඼��40
		//epsilon=0.001
	Mat I;
	cvtColor(image, I, CV_BGR2GRAY);	//ע�������Ƕ�image����ԭͼ����д���
	Point2f p(2.5, 2.5);
	float ratio = 0.1f;
	for (int i=0; i<corner_label.size(); i++)
	{
		//�����ǵ�������2�ı����Ƭ
		if (corner_label[i].size() != 2)
			continue;
		vector<Point2f> temp(2);
		temp[0] = corners_new[corner_label[i][0]];	//ĳ��Ƭ�ϵĵ�һ���ǵ�
		temp[1] = corners_new[corner_label[i][1]];	//ͬһ��Ƭ�ϵĵڶ����ǵ�
		cornerSubPix(image_gray, temp , Size(7,7), Size(-1,-1), criteria);
			//ע�������Ƕ�image_gray����ԭͼ����Сһ����ͼ����������ؾ������
			//���ش���Ϊ��7��2+1������7��2+1����Ҳ����һ������ӳ��Ϊ49��������Q15
			//temp��Ϊ���룬Ҳ��Ϊ���
		temp[0] = temp[0]*SCALE+p;	//��Ϊ֮ǰ�ǽ�ͼ����СΪһ�����õĽǵ����꣬�������Ƕ�image����ԭͼ������жϣ�����Ҫ������Ŵ�һ��������ΪʲôҪ��p�أ�p������ôȷ���ģ�Q16
		temp[1] = temp[1]*SCALE+p;
		cornerSubPix(I, temp , Size(5,5), Size(-1,-1), criteria);	//�ٴ�����������Ϊ֮ǰ�����ز������������С���ͼ�������С��ͼ��������ؽǵ�󣬷Ŵ�һ�����ٽ���һ��������
		//���ֽǵ�˳�򣨿����������ú������� Q17
		Point2f temp_point = (1-ratio)*temp[0]+ratio*temp[1];
		Point2f temp_point2 = (temp[1]-temp[0])*ratio;
		Point2f point_new1(temp_point.x-temp_point2.y, temp_point.y+temp_point2.x);	//���Ͻ�����
		Point2f point_new2(temp_point.x+temp_point2.y, temp_point.y-temp_point2.x);	//���Ͻ�����
		if (I.at<uchar>(point_new1) > I.at<uchar>(point_new2))
		{
			Point2f temp_swap = temp[0];
			temp[0] = temp[1];
			temp[1] = temp_swap;
		}
		pointsImg.push_back(temp);
#ifdef IS_OUTPUT
		circle(I, temp[0], 3, Scalar(255), CV_FILLED);	//IΪԭͼ��ҶȻ����ͼ������ߵ�ΪԲ�ģ��뾶3��Բ������ɫ�����������������ж�temp[0]Ϊ�ҵ�
		circle(I, temp[1], 3, Scalar(0), CV_FILLED);	
#endif
	}
#ifdef IS_OUTPUT
	char filename[512];	//��֣�����˭��filename��ֵ���� Q18 sprintf()������
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
	sphere.at<double>(0,0) = scale*(circles[0]-init_cx);	//Ϊʲôƽ�ƾ����������ģ�T�ķ�����{W}���꣬��ô����ͨ������{I}�������һ�������õ��أ��ǽ����� Q19 �ڼ������ĵ�{W}Ϊ[0,0,0]��ǰ���£�{W}�����ÿ�������ƽ�ƾ�������ô��ģ�˵���ĵ�����˵
	sphere.at<double>(1,0) = scale*(circles[1]-init_cy);
	sphere.at<double>(2,0) = scale*F;
	points3D.resize(pointsImg.size());	//pointsImg.size()��ʾ�����Ƭ������
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
	int src_size = int(points3D_src.size());	//��i��ͼƬ�ı����Ƭ�ĸ���������3��3����ĸ���
	int dst_size = int(points3D_dst.size());	//��i+1��ͼƬ�ı����Ƭ�ĸ�����ע��dst_size��һ������src_size
	int max_inliers = -1, best_i, best_j;
	double threshold = 0.15;
	double min_total_error = 1e6;
	/* ������ʺϵ�R */
	for (int i=0; i<src_size; i++)	//�ĸ�Ƕ��ѭ�������Ӷȣ�src_size^2*dst_size^2�����src_size��dst_size�������൱�����Ӷ�ΪO(n4)������ÿ���ӽ��µı����Ƭ����������̫��
	{
		for (int j=0; j<dst_size; j++)
		{
			//SVD�ֽ���R
			Matx33d test =  points3D_dst[j] * points3D_src[i].t();	//��k+1��ͼƬ�ĵ�j�������Ƭ��Ӧ�ľ��� * ��k��ͼƬ�ĵ�i�������Ƭ��Ӧ�ľ����ת��
			Mat S,U,VT;
			SVD::compute(test,S,U,VT, SVD::FULL_UV);
			Matx33d R_test = Mat(U*VT);	//��src��ת��dst�Ŀ�����ת���󣬼���k�����/{C}����k+1�����/{C}�Ŀ�����ת����
			//ͳ���ڵ�
			int inliers = 0;
			double total_error = 0;
			for (int l=0; l<src_size; l++)	//��src��ʣ��ĵ�Խ��в���
			{
				if (l==i) continue;	//��Ϊ��������R�ĵ�i����ԣ�һ�������������ѭ�����ҵ�һ����ȫ�غϵĵ��j����ômin_errorΪ0����total_errorû��Ӱ�죬����û�б�Ҫ�ټ����ˣ����ڵ㲻������ȫ�غϵĵ㣩
				Matx33d new_src = R_test * points3D_src[l];	//src�е�l�������ת��õ��ĵ�ԣ�������ʽ��
				double min_error = 1e6;
				for (int k=0; k<dst_size; k++)	//�Ҿ�������ĵ�ԣ���min_error��Ӧ�ĵ�ԣ�
				{
					if (k==j) continue;	//����ΪʲôҪ����j�أ� Q20 ��ΪֻҪl����i����ôjһ��������l���ڵ㣨��error��ܴ�
					double error = MAX(norm(new_src.col(0)-points3D_dst[k].col(0)), norm(new_src.col(1)-points3D_dst[k].col(1)));	//Ϊʲô���������������ģ���
						//error����������ģ������������Ƭ���Ե���ߵ�ľ��룬�ұߵ�ľ��룬Ȼ��ȡ���е����ֵ
					if (min_error > error)
						min_error = error;	//��¼��С���
				}
				if (min_error < threshold)	//���С��һ����ֵ������������Ϊ�ڵ�
				{
					inliers++;
					total_error+=min_error;
				}
			}
			if ((inliers == max_inliers && min_total_error > total_error) || inliers > max_inliers )	//���ڲ�ͬ���i��j��õ�������ת����������������ͳ�Ƶõ����ڵ�����ͬ����ȡ�����С��R������ڵ�����ͬ��ȡ�ڵ������R
			{
				min_total_error = total_error;	//����Ŀǰ��С�����������ڵ���������Ӧ��R���Լ��������R�ĵ��i��j
				max_inliers = inliers;
				R = Mat(R_test);
				best_i = i;
				best_j = j;
			}
		}
	}
	if (max_inliers == -1)	//���һ���ڵ㶼û�У��˳�
	{
		printf("no inliers!!!\n");
		exit(0);
	}
	/* �����жϿɼ��ԣ������ӿɼ�������� */
	Mat R_t = R.t();	//A��B����ת����=B��A����ת�����ת��
	Matx33d R_temp = R_t;
	double total_error = 0;
	int points_end_idx = int(point3d_final.size())/2;
		//points_end_index��ʾ�м�¼�ı����Ƭ����ţ�Ҳ���ǿɼ��ı����Ƭ������
		//����int(point3d_final.size())/2����Ϊpoints3d_final.size()���ǽǵ������������2���Ǳ����Ƭ������
		//��ʵ����point3d_final���˻�û�и�ֵ����������sizeΪ0��points_end_indexΪ0���Ҿ���ֱ�Ӹ�ֵΪ0�ͺ���
	visibility_dst.resize(dst_size);
	for (int k=0; k<dst_size; k++)
	{
		if (k==best_j)	//���Ǹ������������������Ϊ�˼��ټ��㣬��ҪҲ����
		{
			if (visibility_src[best_i] == -1)	//�����best_j�����Զ���best_i������û�м�¼���ǾͰ�best_i�����Լ��뵽�ɼ��㼯�У����������points_end_index����ʾ�������۵��ĵ�points_end_index�������Ƭ
			{
				for (int i=0; i<2; i++)
				{
					Mat u = radii*Mat(points3D_src[best_i].col(i));	//u��¼�ľ���best_i�����Ƭ��Ӧ�Ľǵ��{C}�����꣬����֣�points3D_src�洢�ı����Ǿ��󣬾����ÿ�б�ʾһ����λ����������ֱ���ð뾶�˵�λ�����õ��ǵ��{C}���꣬�ǲ�����������������{C}ԭ���� Q21
					Point3d p(u.at<double>(0,0), u.at<double>(1,0), u.at<double>(2,0));
					point3d_final.push_back(p);
				}
				visibility_src[best_i] = points_end_idx;	//��best_i�����Լ��뵽��k������Ŀɼ��㼯�У����������points_end_index����ʾ���ǿ��ü����ĵ�points_end_index�������Ƭ
				points_end_idx++;
			}
			visibility_dst[best_j] = visibility_src[best_i];	//��������ı����Ƭ�����ͬ����ʾ��������ͬһ����Ƭ
			continue;
		}
		//�����￪ʼ���������жϿɼ��Ե������㷨
		Matx33d new_dst = R_temp * points3D_dst[k];
		double min_error = 1e6;
		int fit_l = 0;
		//���n+1������е��k��Ӧ�ĵ�n������еĵ��l������С����ֵ����Ϊ��Ӧ��ԣ�
		for (int l=0; l<src_size; l++)
		{
			//if (l==best_j) continue;	//l==best_j������Ӧ����l==best_i�𣿣� Q22 �Ҿ���Ӧ����l==best_i
			if (l == best_i) continue;
			double error = norm(new_dst.col(0)-points3D_src[l].col(0))+norm(new_dst.col(1)-points3D_src[l].col(1));
				//������붨�岻ͬ��֮ǰ��֮ǰ��������������ֵΪ����������������֮����Ϊ��Ϊʲô�أ� Q23 
				//A23 ��Ӧ���������׼ȷ�ԡ�֮ǰ���ȡ�������������е����ֵ���п�����������ܽӽ������ʱ��ֻ���������ֵ��
				//ֻҪ���ֵС��min_error�Ϳ��Կ����ڵ��ˣ�������ȡ��������ĺͣ��õ���error���ֻ���е����ֵʱ����ô��
				//�͸������ױ������ڵ㣬���������������min_error<threshold����Ϊthresholdǰ��ȡֵ��ͬ��
			if (min_error > error)
			{
				min_error = error;
				fit_l = l;
			}
		}
		if (min_error < threshold)	//����ҵ�k��Զ�Ӧ�ĵ��l�����С����ֵ����Ϊ�ҵ������Ǿͱ��һ����������ͬһ����Ƭ����ʾ�����Ƭ�����ӽ��¶����Կ���
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
		else //���û���ҵ���˵����n+1������ϵ�������k�ڵ�n��������ǿ�������
			visibility_dst[k] = -1;
	}
	for (int k = 0; k < dst_size; k++)
		points3D_dst[k] = R_temp * points3D_dst[k];
			//ע�⣡����Ե�n+1������ĵ�Խ������޸ģ�����
			//������������϶���Щ�ǵ��Ǳ˴˿������ģ���Щ�Ǳ˴˶����Կ����ģ����Կ����Ĳ�������Ҫ���۵ģ�
			//��ôĿ���������ת�������֮����������ԭ�ȱ˴˶����Կ����ĵ㣨�������غ��ˣ��˴˿������ĵ������㣬
			//���Կ���point3d_final push�Ķ���ͨ��pointd3D_src[]����õ��ĵ㣬��Ŀ������ֻ���Ϊ��һ��ѭ����Դ�����
			//��ôҲ����˵������������һ���ӽ�������Ŀɼ�������������ο�������ӽǣ������������ӽǣ�
			//Q24�����������и����⣬Ŀ���������Щ�ο�����������Ľǵ㣬������һ��ѭ���б������ο�����ϵĽǵ������ۣ��ǲ��ʹ�����
			//�Ҿ��ò�Ӧ�ð�Ŀ�������������󣬶�Ӧ��ֱ�ӽ�Ŀ�������ֵΪԴ���������������ı������ݸ�ֵǰ��Ŀ������ǵ㣩
			//����ǰ������Ƭ������points3D_dst[k] = points3D_src[k];���������������ΪĿ������ı����Ƭ���������Ͳ�����Դ����ı����Ƭ������
			//A25 ���ǵģ�points3d_final��¼�������ϴ󲿷ֽǵ㣨֮����˵�󲿷�����Ϊ������Щ�ǵ����û�м�⵽���ڲο��������{C}������
			//RANSAC()ţ�ƣ�
}
			
void CCalibBall::Two_Pass(const Mat& binImg, Mat& lableImg)    //����ɨ�跨
{
	if (binImg.empty() ||
		binImg.type() != CV_8UC1)
	{
		return;
	}

	// ��һ��ͨ·

	lableImg.release();	//��������������������ʱlabelImg��û�г�ʼ������Ӧ������Ϊһ��������һ����
	binImg.convertTo(lableImg, CV_32SC1);

	int label = 1; 
	vector<int> labelSet;
	labelSet.push_back(0);  
	labelSet.push_back(1);  

	int rows = binImg.rows - 1;
	int cols = binImg.cols - 1;
	for (int i = 1; i < rows; i++)	//i = 1 ~ rows-2��why?
	{
		int* data_preRow = lableImg.ptr<int>(i-1);
		int* data_curRow = lableImg.ptr<int>(i);
		for (int j = 1; j < cols; j++)
		{
			if (data_curRow[j] == 1)
			{
				vector<int> neighborLabels;	//��ʱ����neighborLabels��sizeΪ0��capacityΪ0
				neighborLabels.reserve(2);	//��ʱ����neighborLabels��sizeΪ0��capacityΪ2���б�Ҫ��
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
					labelSet.push_back(++label);  // ����ͨ����ǩ+1
					data_curRow[j] = label;
					labelSet[label] = label;
				}
				else
				{
					std::sort(neighborLabels.begin(), neighborLabels.end());
					int smallestLabel = neighborLabels[0];  
					data_curRow[j] = smallestLabel;

					// ������С�ȼ۱�
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

	// ���µȼ۶��б�
	// ����С��Ÿ��ظ�����
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
	int src_size = int(points3D.size());	//��i��ͼƬ�ı����Ƭ�ĸ���
	points.resize(src_size);
	Mat temp(3,3,CV_64FC1);
	for (int i=0; i<src_size; i++)
	{
		Mat(points3D[i][0] * (1/norm(points3D[i][0]))).copyTo(temp.col(0));	//norm(points3D[i][0])���һ����ά���L2��������ŷ����÷���sqrt(x1^2+x2^2+...+xn^2)
		
																			//Mat(const cv::Point3d &ptr)����ά���xyz����д��������
		Mat(points3D[i][1] * (1/norm(points3D[i][1]))).copyTo(temp.col(1));
		Mat(temp.col(1).cross(temp.col(0))).copyTo(temp.col(2));	//cross���������Ǵ�ֱ������a��b��c����������������
		temp.col(2) /= norm(temp.col(2));
		temp.copyTo(points[i]);	//��i�������Ƭ��Ӧ��3��3����
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
