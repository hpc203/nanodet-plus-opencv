#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

typedef struct BoxInfo
{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	int label;
} BoxInfo;

class NanoDet_Plus
{
public:
	NanoDet_Plus(string model_path, string classesFile, int imgsize, float nms_threshold, float objThreshold);
	void detect(Mat& cv_image);
private:
	float score_threshold = 0.5;
	float nms_threshold = 0.5;
	vector<string> class_names;
	int num_class;

	Mat resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left);
	void normalize(Mat& srcimg);
	void softmax_(const float* x, float* y, int length);
	void generate_proposal(vector<BoxInfo>& generate_boxes, const float* preds);
	void nms(vector<BoxInfo>& input_boxes);
	const bool keep_ratio = false;
	int inpWidth;
	int inpHeight;
	const int reg_max = 7;
	const int num_stages = 4;
	const int stride[4] = { 8,16,32,64 };
	const float mean[3] = { 103.53, 116.28, 123.675 };
	const float std[3] = { 57.375, 57.12, 58.395 };
	Net net;
};

NanoDet_Plus::NanoDet_Plus(string model_path, string classesFile, int imgsize, float nms_threshold, float objThreshold)
{
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) this->class_names.push_back(line);
	this->num_class = class_names.size();
	this->nms_threshold = nms_threshold;
	this->score_threshold = objThreshold;

	this->inpHeight = imgsize;
	this->inpWidth = imgsize;
	this->net = readNet(model_path);
}

Mat NanoDet_Plus::resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	Mat dstimg;
	if (this->keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = this->inpHeight;
			*neww = int(this->inpWidth / hw_scale);
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*left = int((this->inpWidth - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, BORDER_CONSTANT, 0);
		}
		else {
			*newh = (int)this->inpHeight * hw_scale;
			*neww = this->inpWidth;
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*top = (int)(this->inpHeight - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, BORDER_CONSTANT, 0);
		}
	}
	else {
		resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
	}
	return dstimg;
}

void NanoDet_Plus::normalize(Mat& img)
{
	img.convertTo(img, CV_32F);
	int i = 0, j = 0;
	for (i = 0; i < img.rows; i++)
	{
		float* pdata = (float*)(img.data + i * img.step);
		for (j = 0; j < img.cols; j++)
		{
			pdata[0] = (pdata[0] - this->mean[0]) / this->std[0];
			pdata[1] = (pdata[1] - this->mean[1]) / this->std[1];
			pdata[2] = (pdata[2] - this->mean[2]) / this->std[2];
			pdata += 3;
		}
	}
}

void NanoDet_Plus::softmax_(const float* x, float* y, int length)
{
	float sum = 0;
	int i = 0;
	for (i = 0; i < length; i++)
	{
		y[i] = exp(x[i]);
		sum += y[i];
	}
	for (i = 0; i < length; i++)
	{
		y[i] /= sum;
	}
}

void NanoDet_Plus::generate_proposal(vector<BoxInfo>& generate_boxes, const float* preds)
{
	const int reg_1max = reg_max + 1;
	const int len = this->num_class + 4 * reg_1max;
	for (int n = 0; n < this->num_stages; n++)
	{
		const int stride_ = this->stride[n];
		const int num_grid_y = (int)ceil((float)this->inpHeight / stride_);
		const int num_grid_x = (int)ceil((float)this->inpWidth / stride_);
		////cout << "num_grid_x=" << num_grid_x << ",num_grid_y=" << num_grid_y << endl;

		for (int i = 0; i < num_grid_y; i++)
		{
			for (int j = 0; j < num_grid_x; j++)
			{
				int max_ind = 0;
				float max_score = 0;
				for (int k = 0; k < num_class; k++)
				{
					if (preds[k] > max_score)
					{
						max_score = preds[k];
						max_ind = k;
					}
				}
				if (max_score >= score_threshold)
				{
					const float* pbox = preds + this->num_class;
					float dis_pred[4];
					float* y = new float[reg_1max];
					for (int k = 0; k < 4; k++)
					{
						softmax_(pbox + k * reg_1max, y, reg_1max);
						float dis = 0.f;
						for (int l = 0; l < reg_1max; l++)
						{
							dis += l * y[l];
						}
						dis_pred[k] = dis * stride_;
					}
					delete[] y;
					/*float pb_cx = (j + 0.5f) * stride_ - 0.5;
					float pb_cy = (i + 0.5f) * stride_ - 0.5;*/
					float pb_cx = j * stride_;
					float pb_cy = i * stride_;
					float x0 = pb_cx - dis_pred[0];
					float y0 = pb_cy - dis_pred[1];
					float x1 = pb_cx + dis_pred[2];
					float y1 = pb_cy + dis_pred[3];
					generate_boxes.push_back(BoxInfo{ x0, y0, x1, y1, max_score, max_ind });
				}
				preds += len;
			}
		}
	}

}

void NanoDet_Plus::nms(vector<BoxInfo>& input_boxes)
{
	sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
	vector<float> vArea(input_boxes.size());
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
			* (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
	}

	vector<bool> isSuppressed(input_boxes.size(), false);
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		if (isSuppressed[i]) { continue; }
		for (int j = i + 1; j < int(input_boxes.size()); ++j)
		{
			if (isSuppressed[j]) { continue; }
			float xx1 = (max)(input_boxes[i].x1, input_boxes[j].x1);
			float yy1 = (max)(input_boxes[i].y1, input_boxes[j].y1);
			float xx2 = (min)(input_boxes[i].x2, input_boxes[j].x2);
			float yy2 = (min)(input_boxes[i].y2, input_boxes[j].y2);

			float w = (max)(float(0), xx2 - xx1 + 1);
			float h = (max)(float(0), yy2 - yy1 + 1);
			float inter = w * h;
			float ovr = inter / (vArea[i] + vArea[j] - inter);

			if (ovr >= this->nms_threshold)
			{
				isSuppressed[j] = true;
			}
		}
	}
	// return post_nms;
	int idx_t = 0;
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &isSuppressed](const BoxInfo& f) { return isSuppressed[idx_t++]; }), input_boxes.end());
}

void NanoDet_Plus::detect(Mat& srcimg)
{
	int newh = 0, neww = 0, top = 0, left = 0;
	Mat cv_image = srcimg.clone();
	Mat dst = this->resize_image(cv_image, &newh, &neww, &top, &left);
	this->normalize(dst);
	Mat blob = blobFromImage(dst);

	this->net.setInput(blob);
	vector<Mat> outs;
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());
	/////generate proposals
	vector<BoxInfo> generate_boxes;
	const float* preds = (float*)outs[0].data;
	generate_proposal(generate_boxes, preds);

	//// Perform non maximum suppression to eliminate redundant overlapping boxes with
	//// lower confidences
	nms(generate_boxes);
	float ratioh = (float)cv_image.rows / newh;
	float ratiow = (float)cv_image.cols / neww;
	for (size_t i = 0; i < generate_boxes.size(); ++i)
	{
		int xmin = (int)max((generate_boxes[i].x1 - left)*ratiow, 0.f);
		int ymin = (int)max((generate_boxes[i].y1 - top)*ratioh, 0.f);
		int xmax = (int)min((generate_boxes[i].x2 - left)*ratiow, (float)cv_image.cols);
		int ymax = (int)min((generate_boxes[i].y2 - top)*ratioh, (float)cv_image.rows);
		rectangle(srcimg, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 0, 255), 2);
		string label = format("%.2f", generate_boxes[i].score);
		label = this->class_names[generate_boxes[i].label] + ":" + label;
		putText(srcimg, label, Point(xmin, ymin - 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
	}
}

int main()
{
	NanoDet_Plus mynet("onnxmodel/nanodet-plus-m_320.onnx", "onnxmodel/coco.names", 320, 0.5, 0.5);  /// choice = ["picodet_m_320_coco.onnx", "picodet_m_416_coco.onnx", "picodet_s_320_coco.onnx", "picodet_s_416_coco.onnx"]
	string imgpath = "imgs/person.jpg";
	Mat srcimg = imread(imgpath);
	mynet.detect(srcimg);

	static const string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();
}