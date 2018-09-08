#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching/stitcher.hpp"
#include <time.h>
#include"mean_shift.h"
#include<iostream>
coordinate::coordinate()
{
	x = 0;
	y = 0;
}

coordinate::coordinate(double xx,double yy)
{
	x = xx;
	y = yy;
}

coordinate::~coordinate()
{
	;
}

double coordinate::get(int pos)
{
	if (pos == 1)
		return x;
	else
		return y;
}


double norm_my(double m1,double m2)
{
	return(sqrt(m1*m1 + m2*m2));
}

vector<int> mean_shift(vector<coordinate>data,int &num_cluster)
{
	vector<int>idx;
	int m = data.size();
	int n = 2;
	vector<int>index;
	for (int i = 0; i < m; i++)
		index.push_back(i+1);
	double radius = 5;//°ë¾¶´óÐ¡
	double stopthresh = 0.0001;
	vector<int>visitflag(m,0);
	//visitflag.assign(m,0);
	vector<vector<int>>count;
	int clustern = 0;
	vector<coordinate>clustercenter;
	while (index.size()>0)
	{
		srand(time(0));
		int cn = index.size()*(rand() % 10000 / (double)10000);
		coordinate center = data[index[cn]];
		vector<int>this_class(m,0);
		
		while (1)
		{
			vector<double>dis(m, 0);
			vector<int> innerS;
			for (int i = 0; i < m; i++)
			{
				dis[i] = pow((center.get(1) - data[i].get(1)), 2) + pow((center.get(2) - data[i].get(2)), 2);
				if (dis[i] < radius*radius)
				{
					visitflag[i] = 1;
					innerS.push_back(i);
					this_class[i]++;
				}
			}
			coordinate newcenter;
			double sum_weight = 0;
			for (int i = 0; i < innerS.size(); i++)
			{
				double w = exp(dis[innerS[i]] / (radius*radius));
				sum_weight += w;
				newcenter = newcenter+ w*data[innerS[i]];
			}
			newcenter = newcenter/sum_weight;
			coordinate temp = newcenter - center;
		//	cout << norm_my(temp.get(1), temp.get(2)) << endl; 
			if (norm_my(temp.get(1),temp.get(2))<stopthresh)
			//if (fabs(newcenter.get(1) - center.get(1)) + fabs(newcenter.get(2) - center.get(2)) < stopthresh)
				break;
			center = newcenter;
			
		}

		int mergewith = -1;
		for (int i = 0; i < clustern; i++)
		{
			coordinate temp;
			temp = center - clustercenter[i];
			double betw = norm_my(temp.get(1), temp.get(2));
			if (betw < radius / 2)
			{
				mergewith = i;
				break;
			}
		}

		if (mergewith == -1)
		{
			clustern++;
			clustercenter.resize(clustern);
			clustercenter[clustern-1]=center;
			count.resize(clustern);
			for (int j = 0; j < this_class.size();j++)
				count[clustern-1].push_back(this_class[j]);
		}
		else
		{
			clustercenter[mergewith ] = 0.5*(clustercenter[mergewith] + center);
			for (int j = 0; j < this_class.size(); j++)
				count[mergewith][j] = count[mergewith][j]+this_class[j];
		}
		vector <int>().swap(index);
		for (int i = 0; i < visitflag.size(); i++)
		{
			if (visitflag[i] == 0)
				index.push_back(i);
		}
	}
	for (int i = 0; i < m; i++)
	{
		int max = -1;
		int index_temp = 0;
		for (int j = 0; j < count.size(); j++)
		{
			if (count[j][i]>max)
			{
				index_temp = j;
				max = count[j][i];
			}
		}
		idx.push_back(index_temp);
	}
	num_cluster = count.size();
	return idx;
}