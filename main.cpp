#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "ImageFeature/ImageFeature.h"


#define IMG_FILE	"1.jpg"
#define WND_NAME_ORIGINAL	"Original"
#define WND_NAME_FEATURES	"Features"

int main()
{
	cv::Mat img = cv::imread(IMG_FILE);
	cv::imshow(WND_NAME_ORIGINAL, img);
	image_feature feature_0(img);
	cv::imshow(WND_NAME_FEATURES, DrawPats({feature_0}));
	cv::waitKey(0);
	std::vector<image_feature> features_1 = ProduceFeatures(feature_0, img);
	for(unsigned i = 0; i < 20; i++)
	{
		cv::imshow(WND_NAME_FEATURES, DrawPats(features_1));
		features_1 = ProduceFeatures(features_1, img);
		if(cv::waitKey(0) == 27)
			break;
	}
	return 0;
}