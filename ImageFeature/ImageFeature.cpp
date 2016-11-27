#include "ImageFeature.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>


image_feature::image_feature(cv::Mat img, cv::Rect rect)
{
	inverse_pat = false;
	pat = PAT_UNKNOWN;
	if(!img.empty())
		Set(img, rect);
}
void image_feature::Set(cv::Mat img, cv::Rect rect)
{
	subrect = rect;
	if((rect.width == 0) || (rect.height == 0))
	{
		subrect.width = img.cols;
		subrect.height = img.rows;
	}
	int pat_ret = GetPat(img(subrect));
	if(pat_ret < 0)
		inverse_pat = true;
	pat = (PAT_TYPE)abs(pat_ret);
	light_limits.resize(2);
	double min_val, max_val;
	cv::Mat img_gray;
	cv::cvtColor(img(subrect), img_gray, CV_BGR2GRAY);
	cv::minMaxLoc(img_gray, &min_val, &max_val);
	light_limits[0] = min_val;
	light_limits[1] = max_val;
	dom_colors = col_grab.GetDomColors(img(subrect));
}

image_features::image_features(cv::Mat img)
{
	_features.resize(HOR_PARTS * VER_PARTS);
	if(!img.empty())
		Set(img);
}
void image_features::Set(cv::Mat img)
{
	cv::Rect roi_rect(0, 0, img.cols/HOR_PARTS, img.rows / VER_PARTS);
	for(unsigned i = 0; i < HOR_PARTS * VER_PARTS; i++)
	{
		roi_rect.x = (i % HOR_PARTS) * roi_rect.width;
		roi_rect.y = (i / HOR_PARTS) * roi_rect.height;
		cv::Mat chunk = img(roi_rect);
		_features[i].Set(chunk);
	}
}
cv::Mat image_features::CompareFile(std::string img_file)
{
	return CompareImg(cv::imread(img_file));
}
cv::Mat image_features::CompareImg(cv::Mat img)
{
	return Compare(image_features(img));
}
cv::Mat image_features::Compare(const image_features& features)
{
	cv::Mat res(VER_PARTS, HOR_PARTS, CV_64FC1);
	for(unsigned i = 0; i < _features.size(); i++)
	{
		res.at<double>(i / HOR_PARTS, i % HOR_PARTS) = 
			cv::compareHist(_features[i].hist, features._features[i].hist, cv::HISTCMP_BHATTACHARYYA);
	}
	return res;
}

double CheckCompareMat(cv::Mat cmp_mat, cv::Mat w)
{
	cv::Mat res_mat = cmp_mat.clone();
	cv::multiply(cmp_mat, w, res_mat);
	cv::Scalar diff = cv::sum(res_mat);
	//diff[0] /= res_mat.cols * res_mat.rows;
#ifdef _DEBUG
	std::cout<< "--------------------------\n";
	std::cout<< cmp_mat <<std::endl;
	std::cout<< w <<std::endl;
	std::cout<< res_mat <<std::endl;
	std::cout<< "Diff: " << diff[0] << std::endl;
#endif
	return diff[0];
}
int GetPat(cv::Mat img)
{
	cv::Mat img_gray;
	if(img.channels() != 1)
		cv::cvtColor(img, img_gray, CV_BGR2GRAY);
	else
		img.copyTo(img_gray);
	cv::threshold(img_gray, img_gray, 0, 255, cv::THRESH_OTSU);
	cv::Mat img_gray_neg = 255 - img_gray;
	int res = PAT_PLAIN;
	double diff = GetPatDiff(img_gray, PAT_PLAIN);
	for(int i = 1; i < patterns.size(); i++)
	{
		double temp = GetPatDiff(img_gray, (PAT_TYPE)i);
		if(temp < 0)
			continue;
		if(temp < diff)
		{
			diff = temp;
			res = i;
		}
		temp = GetPatDiff(img_gray_neg, (PAT_TYPE)i);
		if(temp < 0)
			continue;
		if(temp < diff)
		{
			diff = temp;
			res = -i;
		}
	}
	return res;
}
double GetPatDiff(cv::Mat img, PAT_TYPE pt)
{
	cv::Mat sized_pat;
	if((img.cols < patterns[pt].cols) || (img.rows < patterns[pt].rows))
		return -1;
	cv::resize(patterns[pt], sized_pat, img.size(), 0, 0, cv::INTER_NEAREST);
	cv::Mat diff;
	cv::absdiff(img, sized_pat, diff);
	return cv::sum(diff)[0];
}
std::vector<image_feature> ProduceFeatures(image_feature feature, cv::Mat img, PAT_TYPE pt)
{
	std::vector<image_feature> res;
	if(pt == PAT_UNKNOWN)
		pt = feature.pat;
	std::vector<cv::Rect> rects = GetPatRects(pt, feature.subrect);
	res.resize(rects.size());
	for(unsigned i = 0; i < res.size(); i++)
		res[i].Set(img, rects[i]);
	return res;
}
std::vector<image_feature> ProduceFeatures(std::vector<image_feature> features, cv::Mat img)
{
	std::vector<image_feature> res;
	for(auto& feat: features)
	{
		 std::vector<image_feature> features_temp = ProduceFeatures(feat, img);
		 res.insert(res.end(), features_temp.begin(), features_temp.end());
	}
	return res;
}
std::vector<cv::Rect> GetPatRects(PAT_TYPE pt, cv::Rect parent_rect)
{
	std::vector<cv::Rect> res;
	int hor_last = parent_rect.width;
	int ver_last = parent_rect.height;
	int hor_offset = 0;
	int ver_offset = 0;
	unsigned ver_parts = 1;
	unsigned hor_parts = 1;
	switch(pt)
	{
	case PAT_PLAIN:
	case PAT_NOISE:
	case PAT_DOT:
		ver_parts = 3;
		hor_parts = 3;
		break;
	case PAT_LINE_TOP:
	case PAT_LINE_MIDDLE:
	case PAT_LINE_BOTTOM:
		ver_parts = 3;
		break;
	case PAT_LINE_LEFT:
	case PAT_LINE_CENTER:
	case PAT_LINE_RIGHT:
		hor_parts = 3;
		break;
	case PAT_HALF_H:
		ver_parts = 2;
		break;
	case PAT_HALF_V:
		hor_parts = 2;
		break;
	case PAT_CM:
		ver_parts = 2;
		hor_parts = 2;
		break;
	};
	ver_parts = std::min<int>(ver_parts, parent_rect.height);
	hor_parts = std::min<int>(hor_parts, parent_rect.width);
	res.resize(ver_parts * hor_parts);
	for(unsigned row = 0; row < ver_parts; row++)
	{
		hor_last = parent_rect.width;
		hor_offset = 0;
		for(unsigned col = 0; col < hor_parts; col++)
		{
			int id = row * hor_parts + col;
			res[id].x = hor_offset + parent_rect.x;
			res[id].y = ver_offset + parent_rect.y;
			res[id].width = hor_last / (hor_parts - col);
			res[id].height = ver_last / (ver_parts - row);
			hor_offset += res[id].width;
			hor_last -= res[id].width;
		}
		ver_offset += res[row * hor_parts].height;
		ver_last -= res[row * hor_parts].height;
	}
	return res;
}
cv::Mat DrawPats(std::vector<image_feature> features)
{
	cv::Mat res = cv::Mat(GetOrigSize(features), CV_8UC1);
	for(auto& feature: features)
	{
		DrawPattern(res, feature.subrect, feature.pat, feature.inverse_pat, feature.light_limits);
	}
	return res;
}
cv::Mat DrawDomColors(std::vector<image_feature> features, unsigned colors_count)
{
	cv::Mat res = cv::Mat(GetOrigSize(features), CV_8UC3);
	if(colors_count == 0)
		colors_count = 1;
	for(auto& feature: features)
	{
		cv::rectangle(res, feature.subrect, feature.dom_colors[0], CV_FILLED);
	}
	return res;
}
cv::Size GetOrigSize(std::vector<image_feature>& features)
{
	cv::Size res(0, 0);
	for(auto& feature: features)
	{
		if(res.width < feature.subrect.x + feature.subrect.width)
			res.width = feature.subrect.x + feature.subrect.width;
		if(res.height < feature.subrect.y + feature.subrect.height)
			res.height = feature.subrect.y + feature.subrect.height;
	}
	return res;
}
void DrawPattern(cv::Mat& img, cv::Rect rect, PAT_TYPE pt, bool inverse, std::vector<unsigned char> colors)
{
	cv::Mat img_roi = img(rect);
	cv::Mat resized_pat;
	cv::resize(patterns[pt], resized_pat, rect.size(), 0, 0, cv::INTER_NEAREST);
	if(inverse)
		resized_pat = 255 - resized_pat;
	if(!colors.size())
	{
		colors.resize(2);
		colors[0] = 0;
		colors[1] = 255;
	}
	cv::Mat black_mask;
	cv::inRange(resized_pat, cv::Scalar(0), cv::Scalar(1), black_mask);
	resized_pat.setTo(colors[0], black_mask);
	black_mask = 255 - black_mask;
	resized_pat.setTo(colors[1], black_mask);
	resized_pat.copyTo(img_roi);
}
void DrawColor(cv::Mat& img, cv::Rect rect, cv::Scalar color)
{
	/* for future use*/
}
cv::Mat pat_plain = (cv::Mat_<unsigned char>(1,1) << 0);

cv::Mat pat_top = (cv::Mat_<unsigned char>(3,1) << 0, 255, 255);
cv::Mat pat_middle = (cv::Mat_<unsigned char>(3,1) << 255, 0, 255);
cv::Mat pat_bottom = (cv::Mat_<unsigned char>(3,1) << 255, 255, 0);

cv::Mat pat_left = (cv::Mat_<unsigned char>(1,3) << 0, 255, 255);
cv::Mat pat_center = (cv::Mat_<unsigned char>(1,3) << 255, 0, 255);
cv::Mat pat_right = (cv::Mat_<unsigned char>(1,3) << 255, 255, 0);

cv::Mat pat_h = (cv::Mat_<unsigned char>(2,1) << 255, 0);
cv::Mat pat_v = (cv::Mat_<unsigned char>(1,2) << 0, 255);

cv::Mat pat_cm = (cv::Mat_<unsigned char>(2,2) << 255, 0, 0, 255);

cv::Mat pat_dot = (cv::Mat_<unsigned char>(3,3) << 255, 255, 255, 
													255, 0, 255,
													255, 255, 255);

std::vector<cv::Mat> patterns = {cv::Mat(), pat_plain, 
								pat_top, pat_middle, pat_bottom,
								pat_left, pat_center, pat_right,
								pat_h, pat_v, pat_cm, pat_dot};

dominant_colors_graber col_grab;