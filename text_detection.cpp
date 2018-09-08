#include"iostream"
#include<fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching/stitcher.hpp"
#include <opencv2/features2d/features2d.hpp>
#include<opencv2/nonfree/nonfree.hpp>
#include<opencv2/legacy/legacy.hpp>
#include"mean_shift.h"
#define max_my(a, b)  (((a) > (b)) ? (a) : (b))
#define     NO_OBJECT       0  
#define     MIN(x, y)       (((x) < (y)) ? (x) : (y))  
#define     ELEM(img, r, c) (CV_IMAGE_ELEM(img, unsigned char, r, c))  
#define     ONETWO(L, r, c, col) (L[(r) * (col) + c])  

using namespace std;
using namespace cv;
int num_sift=1000;
vector<double>gray_diff;
vector<size_t>order;


template < typename T>
vector< size_t>  sort_indexes(const vector< T>  & v) {

	// initialize original index locations
	vector< size_t>  idx(v.size());
	for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

	// sort indexes based on comparing values in v
	sort(idx.begin(), idx.end(),
		[&v](size_t i1, size_t i2) {return v[i1] >  v[i2]; });

	return idx;
}

Mat cal_cor(Mat H, Mat point1)
{
	Mat point2(3, 1, CV_64FC1);
	point2.row(0).col(0) = int((H.at<double>(0, 0)*(point1.at<double>(0, 0)) + H.at<double>(0, 1) * point1.at<double>(0, 1) + H.at<double>(0, 2)) / (H.at<double>(2, 0)*point1.at<double>(0, 0) + H.at<double>(2, 1) *point1.at<double>(1, 0) + H.at<double>(2, 2)));
	point2.row(1).col(0) = int((H.at<double>(1, 0)*(point1.at<double>(0, 0)) + H.at<double>(1, 1) * point1.at<double>(0, 1) + H.at<double>(1, 2)) / (H.at<double>(2, 0)*point1.at<double>(0, 0) + H.at<double>(2, 1) *point1.at<double>(1, 0) + H.at<double>(2, 2)));
	point2.row(2).col(0) = 1;
	return point2;
}

double getThreshVal_Otsu_8u(const cv::Mat& _src)
{
	cv::Size size = _src.size();
	if (_src.isContinuous())
	{
		size.width *= size.height;
		size.height = 1;
	}
	const int N = 256;
	int i, j, h[N] = { 0 };
	for (i = 0; i < size.height; i++)
	{
		const uchar* src = _src.data + _src.step*i;
		for (j = 0; j <= size.width - 4; j += 4)
		{
			int v0 = src[j], v1 = src[j + 1];
			h[v0]++; h[v1]++;
			v0 = src[j + 2]; v1 = src[j + 3];
			h[v0]++; h[v1]++;
		}
		for (; j < size.width; j++)
			h[src[j]]++;
	}

	double mu = 0, scale = 1. / (size.width*size.height);
	for (i = 0; i < N; i++)
		mu += i*h[i];

	mu *= scale;
	double mu1 = 0, q1 = 0;
	double max_sigma = 0, max_val = 0;

	for (i = 0; i < N; i++)
	{
		double p_i, q2, mu2, sigma;

		p_i = h[i] * scale;
		mu1 *= q1;
		q1 += p_i;
		q2 = 1. - q1;

		if (std::min(q1, q2) < FLT_EPSILON || std::max(q1, q2) > 1. - FLT_EPSILON)
			continue;

		mu1 = (mu1 + i*p_i) / q1;
		mu2 = (mu - q1*mu1) / q2;
		sigma = q1*q2*(mu1 - mu2)*(mu1 - mu2);
		if (sigma > max_sigma)
		{
			max_sigma = sigma;
			max_val = i;
		}
	}

	return max_val;
}



bool createDetectorDescriptorMatcher(const string& detectorType, const string& descriptorType, const string& matcherType,
	Ptr<FeatureDetector>& featureDetector,
	Ptr<DescriptorExtractor>& descriptorExtractor,
	Ptr<DescriptorMatcher>& descriptorMatcher)
{
	cout << "< Creating feature detector, descriptor extractor and descriptor matcher ..." << endl;
	if (detectorType == "SIFT" || detectorType == "SURF")
		initModule_nonfree();
	featureDetector = FeatureDetector::create(detectorType);
	descriptorExtractor = DescriptorExtractor::create(descriptorType);
	descriptorMatcher = DescriptorMatcher::create(matcherType);
	cout << ">" << endl;
	bool isCreated = !(featureDetector.empty() || descriptorExtractor.empty() || descriptorMatcher.empty());
	if (!isCreated)
		cout << "Can not create feature detector or descriptor extractor or descriptor matcher of given types." << endl << ">" << endl;
	return isCreated;
}

static std::vector<std::vector<cv::Point>> find1(Mat image)
{
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(image,
		contours, // a vector of contours   
		CV_RETR_TREE, // retrieve the external contours  
		CV_CHAIN_APPROX_NONE); // retrieve all pixels of each contours  

	return contours;

}

int find(int set[], int x)
{
	int r = x;
	while (set[r] != r)
		r = set[r];
	return r;
}
//8连通域，暂时没用到
int bwlabel(IplImage* img, int n, int* labels)
{
	if (n != 4 && n != 8)
		n = 4;
	int nr = img->height;
	int nc = img->width;
	int total = nr * nc;
	// results  
	memset(labels, 0, total * sizeof(int));
	int nobj = 0;                               // number of objects found in image  
	// other variables                               
	int* lset = new int[total];   // label table  
	memset(lset, 0, total * sizeof(int));
	int ntable = 0;
	for (int r = 0; r < nr; r++)
	{
		for (int c = 0; c < nc; c++)
		{
			if (ELEM(img, r, c))   // if A is an object  
			{
				// get the neighboring pixels B, C, D, and E  
				int B, C, D, E;
				if (c == 0)
					B = 0;
				else
					B = find(lset, ONETWO(labels, r, c - 1, nc));
				if (r == 0)
					C = 0;
				else
					C = find(lset, ONETWO(labels, r - 1, c, nc));
				if (r == 0 || c == 0)
					D = 0;
				else
					D = find(lset, ONETWO(labels, r - 1, c - 1, nc));
				if (r == 0 || c == nc - 1)
					E = 0;
				else
					E = find(lset, ONETWO(labels, r - 1, c + 1, nc));
				if (n == 4)
				{
					// apply 4 connectedness  
					if (B && C)
					{        // B and C are labeled  
						if (B == C)
							ONETWO(labels, r, c, nc) = B;
						else {
							lset[C] = B;
							ONETWO(labels, r, c, nc) = B;
						}
					}
					else if (B)             // B is object but C is not  
						ONETWO(labels, r, c, nc) = B;
					else if (C)               // C is object but B is not  
						ONETWO(labels, r, c, nc) = C;
					else
					{                      // B, C, D not object - new object  
						//   label and put into table  
						ntable++;
						ONETWO(labels, r, c, nc) = lset[ntable] = ntable;
					}
				}
				else if (n == 6)
				{
					// apply 6 connected ness  
					if (D)                    // D object, copy label and move on  
						ONETWO(labels, r, c, nc) = D;
					else if (B && C)
					{        // B and C are labeled  
						if (B == C)
							ONETWO(labels, r, c, nc) = B;
						else
						{
							int tlabel = MIN(B, C);
							lset[B] = tlabel;
							lset[C] = tlabel;
							ONETWO(labels, r, c, nc) = tlabel;
						}
					}
					else if (B)             // B is object but C is not  
						ONETWO(labels, r, c, nc) = B;
					else if (C)               // C is object but B is not  
						ONETWO(labels, r, c, nc) = C;
					else
					{                      // B, C, D not object - new object  
						//   label and put into table  
						ntable++;
						ONETWO(labels, r, c, nc) = lset[ntable] = ntable;
					}
				}
				else if (n == 8)
				{
					// apply 8 connectedness  
					if (B || C || D || E)
					{
						int tlabel = B;
						if (B)
							tlabel = B;
						else if (C)
							tlabel = C;
						else if (D)
							tlabel = D;
						else if (E)
							tlabel = E;
						ONETWO(labels, r, c, nc) = tlabel;
						if (B && B != tlabel)
							lset[B] = tlabel;
						if (C && C != tlabel)
							lset[C] = tlabel;
						if (D && D != tlabel)
							lset[D] = tlabel;
						if (E && E != tlabel)
							lset[E] = tlabel;
					}
					else
					{
						//   label and put into table  
						ntable++;
						ONETWO(labels, r, c, nc) = lset[ntable] = ntable;
					}
				}
			}
			else
			{
				ONETWO(labels, r, c, nc) = NO_OBJECT;      // A is not an object so leave it  
			}
		}
	}
	// consolidate component table  
	for (int i = 0; i <= ntable; i++)
		lset[i] = find(lset, i);
	// run image through the look-up table  
	for (int r = 0; r < nr; r++)
	for (int c = 0; c < nc; c++)
		ONETWO(labels, r, c, nc) = lset[ONETWO(labels, r, c, nc)];
	// count up the objects in the image  
	for (int i = 0; i <= ntable; i++)
		lset[i] = 0;
	for (int r = 0; r < nr; r++)
	for (int c = 0; c < nc; c++)
		lset[ONETWO(labels, r, c, nc)]++;
	// number the objects from 1 through n objects  
	nobj = 0;
	lset[0] = 0;
	for (int i = 1; i <= ntable; i++)
	if (lset[i] > 0)
		lset[i] = ++nobj;
	// run through the look-up table again  
	for (int r = 0; r < nr; r++)
	for (int c = 0; c < nc; c++)
		ONETWO(labels, r, c, nc) = lset[ONETWO(labels, r, c, nc)];
	//  
	delete[] lset;
	return nobj;
}

void gray_projection( vector<int> &Sx,vector<int>&Sy, Mat I)
{ 
	int m = I.rows;
	int n = I.cols;
	for (int i = 0; i < m;i++)
	for (int j = 0; j < n;j++)
		Sx[i] +=I.at<uchar>(i, j); 
	for (int i = 0; i < n; i++)
	for (int j = 0; j < m; j++)
		Sy[i] += I.at<uchar>(j,i);


}

int calculate_min(vector<double>v)
{
	double min = INT_MAX;
	int index = -1;
	for (int i = 0; i < v.size(); i++)
	{
		if (min>v[i])
		{
			index = i;
			min = v[i];
		}
	}
	return index;
}

double sum_diff(Mat src, Mat dst)
{
	//src = hist_extend(src);
	//dst = hist_extend(dst);
	double mean1 = 0;
	double mean2 = 0;

	for (int i = 0; i < src.size().height; i++)
		for (int j = 0; j < src.size().width; j++)
		{
		mean1 += double(src.at<uchar>(i, j));
		mean2 += double(dst.at<uchar>(i, j));
		}
	mean1 = mean1 / (src.size().width*src.size().height);
	mean2 = mean2 / (src.size().width*src.size().height);
	double sum = 0;
	for (int i = 0; i < src.size().height; i++)
		for (int j = 0; j< src.size().width; j++)
		{
		sum += abs(double(src.at<uchar>(i, j)) - mean1 - double(dst.at<uchar>(i, j)) + mean2);
		}
	sum = sum / (src.size().height*src.size().width);
	return sum;
}

Mat hist_extend(Mat src, int &flag, int range)
{
	int min = INT_MAX;
	int max = 0;
	Mat dst = src.clone();
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{
		if (min>src.at<uchar>(i, j))
			min = src.at<uchar>(i, j);
		if (max<src.at<uchar>(i, j))
			max = src.at<uchar>(i, j);
		}
	if (max - min > 127 || max - min<10)
		flag = 1;
	else
		flag = 0;
	//cout << max - min << endl;
	if (max - min <= 200 && max - min >= 10)
	{
		for (int i = 0; i < src.rows; i++)
			for (int j = 0; j < src.cols; j++)
			{
			//dst.row(i).col(j) = MIN(int(double(src.at<uchar>(i, j) - min)*2),255);// / double(max - min) * 255);
			dst.row(i).col(j) = int((double(src.at<uchar>(i, j)) - min) / double(max - min)*range);
			}
	}
	return dst;
}

void copy_im(Mat &src_result, int x, int y, int h, int w, int gray_diff_xx2, int gray_diff_yy2, Mat src_select, Mat dst_select)
{

	Mat  result = src_select.rowRange(y, y + h).colRange(x, x + w).clone();
	Mat src_temp = src_select.rowRange(y, y + h).colRange(x, x + w).clone();
	Mat dst_temp = dst_select.rowRange(gray_diff_yy2, gray_diff_yy2 + h).colRange(gray_diff_xx2, gray_diff_xx2 + w).clone();
	double mean1 = 0;
	double mean2 = 0;

	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++)
		{
		mean1 += double(src_temp.at<uchar>(i, j));
		mean2 += double(dst_temp.at<uchar>(i, j));
		}
	mean1 = mean1 / (h*w);
	mean2 = mean2 / (h*w);
	double max = 0;

	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++)
		{
		if (max <abs(double(src_temp.at<uchar>(i, j)) - mean1 - double(dst_temp.at<uchar>(i, j)) + mean2))
			max = abs(double(src_temp.at<uchar>(i, j)) - mean1 - double(dst_temp.at<uchar>(i, j)) + mean2);
		}
	int min_src = INT_MAX, max_src = 0, min_dst = INT_MAX, max_dst = 0;
	for (int i = 0; i < src_temp.rows; i++)
		for (int j = 0; j < src_temp.cols; j++)
		{
		if (min_src>src_temp.at<uchar>(i, j))
			min_src = src_temp.at<uchar>(i, j);
		if (max_src<src_temp.at<uchar>(i, j))
			max_src = src_temp.at<uchar>(i, j);
		}
	for (int i = 0; i < src_temp.rows; i++)
		for (int j = 0; j < src_temp.cols; j++)
		{
		if (min_dst>dst_temp.at<uchar>(i, j))
			min_dst = dst_temp.at<uchar>(i, j);
		if (max_dst<dst_temp.at<uchar>(i, j))
			max_dst = dst_temp.at<uchar>(i, j);
		}
	int range = MAX(max_src - min_src, max_dst - min_dst);

	int flag1 = 0, flag2 = 0;

	while (max > 30 && flag1*flag2 == 0)
	{
		range = MIN(range * 2, 255);
		src_temp = hist_extend(src_temp, flag1, range);
		dst_temp = hist_extend(dst_temp, flag2, range);
		mean1 = 0;
		mean2 = 0;

		for (int i = 0; i < h; i++)
			for (int j = 0; j < w; j++)
			{
			mean1 += double(src_temp.at<uchar>(i, j));
			mean2 += double(dst_temp.at<uchar>(i, j));
			}
		mean1 = mean1 / (h*w);
		mean2 = mean2 / (h*w);
		double max = 0;

		for (int i = 0; i < h; i++)
			for (int j = 0; j < w; j++)
			{
			if (max <abs(double(src_temp.at<uchar>(i, j)) - mean1 - double(dst_temp.at<uchar>(i, j)) + mean2))
				max = abs(double(src_temp.at<uchar>(i, j)) - mean1 - double(dst_temp.at<uchar>(i, j)) + mean2);
			}
	}
	mean1 = 0;
	mean2 = 0;

	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++)
		{
		mean1 += double(src_temp.at<uchar>(i, j));
		mean2 += double(dst_temp.at<uchar>(i, j));
		}
	mean1 = mean1 / (h*w);
	mean2 = mean2 / (h*w);
	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++)
		{
		result.row(i).col(j) = abs(double(src_temp.at<uchar>(i, j)) - mean1 - double(dst_temp.at<uchar>(i, j)) + mean2);
		}
	result.copyTo(src_result.rowRange(y, y + h).colRange(x, x + w));
}

int main()
{


	Ptr<DescriptorMatcher> descriptor_matcher;
	string detectorType = "SIFT";
	string descriptorType = "SIFT";
	string matcherType = "FlannBased";
	Ptr<FeatureDetector> featureDetector;
	Ptr<DescriptorExtractor> descriptorExtractor;
	if (!createDetectorDescriptorMatcher(detectorType, descriptorType, matcherType, featureDetector, descriptorExtractor, descriptor_matcher))
	{
		cout << "Creat Detector Descriptor Matcher False!" << endl;
		return -1;
	}  // = DescriptorMatcher;//::create("BruteForce");
	Mat src = imread("src0.jpg");
	Mat dst = imread("dst0.jpg");
	Mat dst_resize, src_resize,src_result;
	cvtColor(src, src_result, CV_BGR2GRAY);
	//20171030 new-add
	Mat src_select, dst_select;
	cvtColor(src, src_select, CV_BGR2GRAY);
	cvtColor(dst, dst_select, CV_BGR2GRAY);
	GaussianBlur(src_select, src_select, Size(3, 3), 0, 0);
	GaussianBlur(dst_select, dst_select, Size(3, 3), 0, 0);
	//GaussianBlur(src, src, Size(3, 3), 0, 0);
	//GaussianBlur(dst, dst, Size(3, 3), 0, 0);
	//double src_mean = compute_mean(src_select);
	//double dst_mean = compute_mean(dst_select);
	//cout << "src_mean=" << src_mean << endl;;
	//cout << "dst_mean=" << dst_mean << endl;
	vector<KeyPoint>kp1, kp2;

	int maxcorners = 5000;
	double qualitylevel = 0.01;
	double mindistance = 5;
	Mat mask1(src_select.size(),CV_8U,Scalar(255));
		
	Mat	mask2(src_select.size(), CV_8U, Scalar(255));
	
	GoodFeaturesToTrackDetector detector(maxcorners, qualitylevel, mindistance);
	
	detector.detect(src_select, kp1, mask1);
	detector.detect(dst_select, kp2, mask2);

	FREAK descriptor_extractor;

	cout << "kp1_size00=" << kp1.size() << endl;
	cout << "kp2_size00==" << kp2.size() << endl;
	//Ptr<DescriptorExtractor> descriptor_extractor = DescriptorExtractor::create("SIFT");//´´½¨ÌØÕ÷ÏòÁ¿Éú³ÉÆ÷   
	Mat descriptor1, descriptor2;
	descriptor_extractor.compute(src_select, kp1, descriptor1);
	descriptor_extractor.compute(dst_select, kp2, descriptor2);
	vector<DMatch> matches;
	BruteForceMatcher<HammingLUT>matcher;
	matcher.match(descriptor1, descriptor2, matches);


	cout << "match_size=" << matches.size() << endl;
	Mat img_matches;
	drawMatches(src_select, kp1, dst_select, kp2, matches, img_matches);
	Mat img_matches_resize;
	resize(img_matches, img_matches_resize, Size(1600, 900), 0, 0, CV_INTER_LINEAR);
	/*imshow("matches", img_matches_resize);
	cvWaitKey(0);*/

	vector<Point2d>point_src, point_dst;
	for (int i = 0; i < matches.size(); i++)
	{
		Point2d pt_temp_train, pt_temp_query;
		pt_temp_query.x = kp1[matches[i].queryIdx].pt.x;
		pt_temp_query.y = kp1[matches[i].queryIdx].pt.y;
		pt_temp_train.x = kp2[matches[i].trainIdx].pt.x;
		pt_temp_train.y = kp2[matches[i].trainIdx].pt.y;
		point_src.push_back(pt_temp_query);
		point_dst.push_back(pt_temp_train);
	}
	/*dwRetCode = VikeyUserLogin(Index, UserPassWord);
	if (dwRetCode)
	{
	printf("\nERROR: No Permission to Use the Software! \n");
	return -1;
	}
	else
	{
	cout << "vikey success!";
	}*/
	
	Mat T = findHomography(point_src, point_dst, CV_RANSAC);
	cout << T << endl;
	Mat point1(3, 1, CV_64FC1);
	vector<double>diff;
	vector<coordinate>data;
	vector<int>index;

	//ofstream myfile("data.txt", ios::out);


	for (int i = 0; i < point_src.size(); i++)
	{
		point1.row(0).col(0) = point_src[i].x;
		point1.row(1).col(0) = point_src[i].y;
		point1.row(2).col(0) = 1.0;
		Mat point2 = T * point1;
		double x2 = point2.at<double>(0, 0);
		double y2 = point2.at<double>(1, 0);
		if (sqrt(pow(x2 - point_dst[i].x, 2) + pow(y2 - point_dst[i].y, 2)) < 300)
		{
			//myfile << point_dst[i].x - x2 << " " << point_dst[i].y - y2 << endl;
			coordinate c(point_dst[i].x - x2, point_dst[i].y - y2);
			data.push_back(c);
			index.push_back(i);
		}

	}
	int num_cluster = 0;
	vector<int>idx = mean_shift(data, num_cluster);
	vector<int>cluster_number(num_cluster, 0);
	for (int i = 0; i < idx.size(); i++)
	{
		cluster_number[idx[i]]++;
	}

	vector<int>cluster_final;
	for (int i = 0; i < num_cluster; i++)
	{
		if (cluster_number[i]>100)
			//if (cluster_number[i]>idx.size() / (num_cluster * 1.1) && cluster_number[i]>10)
			cluster_final.push_back(i);
	}

	vector<Mat>TT;
	vector<double>center_x;
	vector<double>center_y;
	Mat img_merge;
	Mat outImg_left, outImg_right;
	Size size(src.cols + dst.cols, src.rows);
	img_merge.create(size, CV_MAKETYPE(src.depth(), 3));
	img_merge = Scalar::all(0);
	outImg_left = img_merge(Rect(0, 0, src.cols, src.rows));
	outImg_right = img_merge(Rect(src.cols, 0, dst.cols, dst.rows));
	src.copyTo(outImg_left);
	dst.copyTo(outImg_right);
	//Mat img_matches_resize;
	for (int i = 0; i < cluster_final.size(); i++)
	{
		stringstream stream;
		stream << i;
		string string_temp = stream.str();

		//ofstream myfile_ransac1(string_temp+"src.txt", ios::out);
		//ofstream myfile_ransac2(string_temp +"dst.txt", ios::out);
		Mat img_merge_temp = img_merge.clone();
		vector<Point2f >point_src_temp, point_dst_temp;
		double x_sum = 0, y_sum = 0;
		for (int j = 0; j < idx.size(); j++)
		{
			if (idx[j] == cluster_final[i])
			{
				point_src_temp.push_back(point_src[index[j]]);
				point_dst_temp.push_back(point_dst[index[j]]);
				x_sum = x_sum + point_src[index[j]].x;
				y_sum = y_sum + point_src[index[j]].y;
				//myfile_ransac1 << point_src[index[j]].x << " " << point_src[index[j]].y << endl;
				//myfile_ransac2 << point_dst[index[j]].x << " " << point_dst[index[j]].y << endl;
				//	cout << point_src[index[j]].x - point_dst[index[j]].x << "    " << point_src[index[j]].y - point_dst[index[j]].y << endl;
				Point dst = Point(point_dst[index[j]].x + src.cols, point_dst[index[j]].y);
				line(img_merge_temp, point_src[index[j]], dst, Scalar(255, 0, 255));

				//resize(img_merge_temp, img_matches_resize, Size(1600, 900), 0, 0, CV_INTER_LINEAR);
				//imwrite("matches.jpg", img_matches_resize);
				//cvWaitKey(0);
			}
		}

		imwrite("temp.jpg", img_merge_temp);
		//imshow(string_temp, img_matches_resize);
		//waitKey(0);
		Mat T_temp = findHomography(point_src_temp, point_dst_temp, CV_RANSAC);
		TT.push_back(T_temp);
		//cout << T_temp << endl;
	}
		double parameter_kernel = 50;
		int block_h = src.size().height / parameter_kernel;// int(double(MIN(src.size().height, src.size().width)) / 1000 * 30);
		int block_w = src.size().width / parameter_kernel;

		
		for (int i = 0; i <= src.size().height - block_h; i = i + block_h)
			for (int j = 0; j <= src.size().width - block_w; j = j + block_w)
			{
			
				double x = j;
				double y = i;
				double w = block_w;
				double h = block_h;
				Mat src_temp = src_select.rowRange(y, y + h).colRange(x, x + w).clone();
				double centerx = x + w / 2;
				double centery = y + h / 2;

				int index_min = -1;
				double min = INT_MAX;
				for (int j = 0; j < idx.size(); j++)
				{
					double temp = sqrt(pow(point_src[index[j]].x - centerx, 2) + pow(point_src[index[j]].y - centery, 2));
					if (temp < min)
					{
						int flag = -1;
						for (int p = 0; p < cluster_final.size(); p++)
							if (cluster_final[p] == idx[j])
								flag = p;
						if (flag != -1)
						{
							min = temp;
							index_min = flag;
						}
					}
				}
				
				point1.row(0).col(0) = x;
				point1.row(1).col(0) = y;
				point1.row(2).col(0) = 1.0;
				//Mat	point2 = TT[index_min] * point1;
				//Mat	point2 = T * point1;
				Mat point2 = cal_cor(T, point1);
				//Mat	point2 = point1;
				double x2 = MIN(MAX(point2.at<double>(0, 0), 0), dst.size().width - w);
				double y2 = MIN(MAX(point2.at<double>(1, 0), 0), dst.size().height - h);
				int drift_x = 3; //int(MAX(3, MIN(0.3*w, 5)));
				int drift_y = 3;// int(MAX(3, MIN(0.3*h, 5)));
				
				if (x2 >= 0 && y2 >= 0 && x2 + w <= dst.size().width && y2 + h <= dst.size().height)
				{
					vector<double>gray_diff_min;
					vector<int>gray_diff_xx2;
					vector<int>gray_diff_yy2;
					for (int xx2 = x2 - drift_x; xx2 <= x2 + drift_x; xx2 = xx2 + 1)
						for (int yy2 = y2 - drift_y; yy2 <= y2 + drift_y; yy2 = yy2 + 1)
						{
						if (xx2 >= 0 && yy2 >= 0 && xx2 + w <= src.size().width && yy2 + h <= src.size().height)
						{
							Mat dst_temp = dst_select.rowRange(max_my(yy2, 0), max_my(yy2, 0) + h).colRange(max_my(xx2, 0), max_my(xx2, 0) + w).clone();

							gray_diff_min.push_back(sum_diff(src_temp, dst_temp));
							gray_diff_xx2.push_back(xx2);
							gray_diff_yy2.push_back(yy2);
						}
						}
					int index_min = calculate_min(gray_diff_min);
					if (index_min != -1)
					{
						gray_diff.push_back(gray_diff_min[index_min]);
						//cocy_rgb(src_result, x, y, h, w, gray_diff_xx2[index_min], gray_diff_yy2[index_min], src, dst);
						copy_im(src_result, x, y, h, w, gray_diff_xx2[index_min], gray_diff_yy2[index_min], src_select, dst_select);
					}
				}
		

	       }
		imwrite("1.jpg", src_result);

		//20180050_new_modified
		vector<Mat>T_modified;
		Mat dst_new = src_select.clone();
		int block_num = 10;
		ofstream file("1.txt");
		ofstream file2("matrix.txt");
		
		for (int i = 0, q=0; i < src.size().height, q<block_num; i = i + floor(src.size().height/block_num),q++)
			for (int j = 0, p=0; j < src.size().width, p<block_num; j = j + floor(src.size().width / block_num),p++)
			{
				int x = j;
				int y = i;
					
				int h = floor(src.size().height / block_num);
				int w = floor(src.size().width / block_num);
				cout << y << "  " << y+h <<"   "<<x<<"  "<<x+w<< endl;
				double start = static_cast<double>(getTickCount());
				Mat src_temp_whole = src_select.clone();
			    Mat dst_temp_whole =dst_select.clone();
				src_temp_whole.setTo(0);
				dst_temp_whole.setTo(0);
				
				//cout << w << "    " << h << endl;
				Mat src_temp = src_select.rowRange(y, y + h).colRange(x, x + w).clone();
				int y_start = MAX(y - 0.5*h, 0);
				int y_end = MIN(y + 1.5*h, src_select.size().height);
				int x_start = MAX(x - 0.5* w, 0);
				int x_end = MIN(x + 1.5*w, src_select.size().width);
				Mat dst_temp = dst_select.rowRange(y_start, y_end).colRange(x_start, x_end).clone();

	            vector<KeyPoint>kp11, kp22;
				//src_temp.copyTo(src_temp_whole.rowRange(y, y + h).colRange(x, x + w));
				//dst_temp.copyTo(dst_temp_whole.rowRange(y, y + h).colRange(x, x + w));
				int maxcorners = 500;
				double qualitylevel = 0.01;
				double mindistance = 2;
				//Mat mask1(src_select.size(), CV_8U, Scalar(0));
				//Mat mask2(src_select.size(), CV_8U, Scalar(0));
				Mat mask1, mask2;
				//rectangle(mask1, Point(x, y), Point(x+w, y+h), Scalar(255), -1, CV_8U);
				//rectangle(mask2, Point(x, y), Point(x+w, y+h), Scalar(255), -1, CV_8U);
				GoodFeaturesToTrackDetector detector1(maxcorners, qualitylevel, mindistance);
				//SiftFeatureDetector  detector1(500, 3, 0.01, 10, 1.6);
				//SiftFeatureDetector  detector2(500, 3, 0.01, 10, 1.6);
				detector1.detect(src_temp, kp11);
				detector1.detect(dst_temp, kp22);
				file << kp22.size() << endl;
				double time = ((double)getTickCount() - start) / getTickFrequency();
				cout << "所用时间111111为：" << time << "秒" << endl;

				vector<Point2d>point_src, point_dst;
				if (kp11.size() > 10 && kp22.size() > 10)
				{



					FREAK descriptor_extractor;
					//Ptr<DescriptorExtractor> descriptor_extractor = DescriptorExtractor::create("SIFT");
					Mat descriptor11, descriptor22;
				
					descriptor_extractor.compute(src_temp, kp11, descriptor11);
					descriptor_extractor.compute(dst_temp, kp22, descriptor22);
					
					
					/*vector<vector<DMatch>> knnMatches;
					vector<DMatch> matches;
					descriptor_matcher->knnMatch(descriptor11, descriptor22, knnMatches, 2);
					const float minRatio = 1.f / 1.2f;
					int num = 0;
					for (size_t i = 0; i < knnMatches.size(); i++)
					{
						cv::DMatch& bestMatch = knnMatches[i][0];
						cv::DMatch& betterMatch = knnMatches[i][1];
						float distanceRatio = bestMatch.distance / betterMatch.distance;
						if (distanceRatio < minRatio)
						{
							matches.push_back(bestMatch);
							num++;

						}
					}
					cout << num << endl;*/

				

					vector<DMatch> matches;
					BruteForceMatcher<HammingLUT>matcher;
					matcher.match(descriptor11, descriptor22, matches);	
					/*Mat img_matches;
					drawMatches(src_temp, kp11, dst_temp, kp22, matches, img_matches);
					Mat img_matches_resize;
					resize(img_matches, img_matches_resize, Size(600, 400), 0, 0, CV_INTER_LINEAR);
					imshow("matches", img_matches_resize);
					waitKey(0);*/
					for (int i = 0; i < matches.size(); i++)
					{
						Point2d pt_temp_train, pt_temp_query;
						pt_temp_query.x = kp11[matches[i].queryIdx].pt.x;
						pt_temp_query.y = kp11[matches[i].queryIdx].pt.y;
						pt_temp_train.x = kp22[matches[i].trainIdx].pt.x;
						pt_temp_train.y = kp22[matches[i].trainIdx].pt.y;
						point_src.push_back(pt_temp_query);
						point_dst.push_back(pt_temp_train);
					}
					
					//cvWaitKey(0);
				}
				if (point_src.size() < 10 || point_dst.size() < 10)
				{
					T_modified.push_back(T);
					
				//	src_temp.copyTo(dst_new.rowRange(y, y + h).colRange(x, x + w));
				}
				else
				{
					double start = static_cast<double>(getTickCount());
					cout << point_src.size() << "  " << point_dst.size() << endl;
					Mat TT = findHomography(point_src, point_dst, CV_RANSAC);
					Mat dst_convert_src = src_temp.clone();
					//cout << T << endl;
					T_modified.push_back(TT);
					double time = ((double)getTickCount() - start) / getTickFrequency();
					cout << "所用时间22222为：" << time << "秒" << endl;
					//cout << dst_convert_src.rows << "  " << dst_convert_src.cols << endl;
					warpPerspective(src_temp, dst_convert_src, TT, cv::Size(dst_convert_src.cols, dst_convert_src.rows));

					dst_convert_src.copyTo(dst_new.rowRange(y, y + h).colRange(x, x + w));
					
				}
			}
			
		imwrite("dst_new.jpg", dst_new);
		file.close();
		Mat src_result_add = src_result.clone();
		for (int i = 0; i <= src.size().height - block_h; i = i + block_h)
			for (int j = 0; j <= src.size().width - block_w; j = j + block_w)
			{

			double x = j;
			double y = i;
			double w = block_w;
			double h = block_h;
			Mat src_temp = src_select.rowRange(y, y + h).colRange(x, x + w).clone();
			
			int x_index = int(x + block_w / 2) / floor(src.size().width / block_num);
			int y_index = int(y + block_h / 2) / floor(src.size().height / block_num);

			int index = y_index*block_num + x_index;
			cout << index << endl;
			point1.row(0).col(0) = x;
			point1.row(1).col(0) = y;
			point1.row(2).col(0) = 1.0;
			//Mat	point2 =  point1;
			Mat point2 = cal_cor(T_modified[index], point1);
			
			double x2 = MIN(MAX(point2.at<double>(0, 0), 0), dst.size().width - w);
			double y2 = MIN(MAX(point2.at<double>(1, 0), 0), dst.size().height - h);
			int drift_x = 3; //int(MAX(3, MIN(0.3*w, 5)));
			int drift_y = 3;// int(MAX(3, MIN(0.3*h, 5)));

			if (x2 >= 0 && y2 >= 0 && x2 + w <= dst.size().width && y2 + h <= dst.size().height)
			{
				vector<double>gray_diff_min;
				vector<int>gray_diff_xx2;
				vector<int>gray_diff_yy2;
				for (int xx2 = x2 - drift_x; xx2 <= x2 + drift_x; xx2 = xx2 + 1)
					for (int yy2 = y2 - drift_y; yy2 <= y2 + drift_y; yy2 = yy2 + 1)
					{
					if (xx2 >= 0 && yy2 >= 0 && xx2 + w <= src.size().width && yy2 + h <= src.size().height)
					{
						Mat dst_temp = dst_select.rowRange(max_my(yy2, 0), max_my(yy2, 0) + h).colRange(max_my(xx2, 0), max_my(xx2, 0) + w).clone();

						gray_diff_min.push_back(sum_diff(src_temp, dst_temp));
						gray_diff_xx2.push_back(xx2);
						gray_diff_yy2.push_back(yy2);
					}
					}
				int index_min = calculate_min(gray_diff_min);
				if (index_min != -1)
				{
					gray_diff.push_back(gray_diff_min[index_min]);
					//cocy_rgb(src_result, x, y, h, w, gray_diff_xx2[index_min], gray_diff_yy2[index_min], src, dst);
					copy_im(src_result_add, x, y, h, w, gray_diff_xx2[index_min], gray_diff_yy2[index_min], src_select, dst_select);
				}
			}


			}

		imwrite("2.jpg",src_result_add);
		for (int i = 0; i < src_result.size().height; i++)
			for (int j = 0; j < src_result.size().width; j++)
			{
			if (src_result_add.at<uchar>(i, j) < src_result.at<uchar>(i, j))
				src_result.row(i).col(j) = src_result_add.at<uchar>(i, j);
			}
		imwrite("dst_final.jpg", src_result);


		
	return 0;
}


