#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include "../DomColors/DomColors.h"

#define HOR_PARTS	3
#define VER_PARTS	3

enum PAT_TYPE
{
	PAT_UNKNOWN = 0,
	PAT_NOISE = 0,
	PAT_PLAIN,

	PAT_LINE_TOP, // -
	PAT_LINE_MIDDLE,
	PAT_LINE_BOTTOM,

	PAT_LINE_LEFT, // |
	PAT_LINE_CENTER,
	PAT_LINE_RIGHT,

	PAT_HALF_H, // bottom half is black
	PAT_HALF_V, // left half is black

	PAT_CM,		// 1 0 
				// 0 1
	PAT_DOT,
	PAT_MAX
};

struct image_feature
{
	image_feature(cv::Mat img = cv::Mat(), cv::Rect rect = cv::Rect());
	void Set(cv::Mat img, cv::Rect rect = cv::Rect());
	std::vector<cv::Scalar> dom_colors;
	cv::Mat hist;
	PAT_TYPE pat;
	bool inverse_pat;
	cv::Rect subrect;
	std::vector<unsigned char> light_limits;
};

class image_features
{
public:
	image_features(cv::Mat img = cv::Mat());
	void Set(cv::Mat img);
	cv::Mat CompareFile(std::string img_file);
	cv::Mat CompareImg(cv::Mat img);
	cv::Mat Compare(const image_features& features);
protected:
	std::vector<image_feature> _features;
};

double CheckCompareMat(cv::Mat cmp_mat, cv::Mat w);
int GetPat(cv::Mat img);
double GetPatDiff(cv::Mat img, PAT_TYPE pt);
std::vector<image_feature> ProduceFeatures(image_feature feature, cv::Mat img, PAT_TYPE pt = PAT_UNKNOWN);
std::vector<image_feature> ProduceFeatures(std::vector<image_feature> features, cv::Mat img);
std::vector<cv::Rect> GetPatRects(PAT_TYPE pt, cv::Rect parent_rect);
cv::Mat DrawPats(std::vector<image_feature> features);
cv::Mat DrawDomColors(std::vector<image_feature> features, unsigned colors_count = 1);
cv::Size GetOrigSize(std::vector<image_feature>& features);
void DrawPattern(cv::Mat& img, cv::Rect rect, PAT_TYPE pt, bool inverse, std::vector<unsigned char> colors = std::vector<unsigned char>());
void DrawColor(cv::Mat& img, cv::Rect rect, cv::Scalar color);
extern std::vector<cv::Mat> patterns;
extern dominant_colors_graber col_grab;