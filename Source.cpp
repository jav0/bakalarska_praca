/*
* textdetection.cpp
*
* A demo program of End-to-end Scene Text Detection and Recognition:
* Shows the use of the Tesseract OCR API with the Extremal Region Filter algorithm described in:
* Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012
*
* Created on: Jul 31, 2014
*     Author: Lluis Gomez i Bigorda <lgomez AT cvc.uab.es>
*/

#include "opencv2/text.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"

#include "FontDatabaseLoader.h"
#include "Tester.h"
#include "DataStructures.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
using namespace cv;
using namespace cv::text;
using namespace cv::xfeatures2d;


// Main function recognizing fonts
void FontRecognition(int testCase = -1);
//Calculate edit distance netween two words
size_t min(size_t x, size_t y, size_t z);
bool   isRepetitive(const string& s);
bool   sort_by_lenght(const string &a, const string &b);
//Draw ER's in an image via floodFill
void   er_draw(vector<Mat> &channels, vector<vector<ERStat> > &regions, vector<Vec2i> group, Mat& segmentation);
//sorts letters in word by x coord in Rect from left to right
void sortt(vector<Rect> &rs, vector<Mat> &gr);
//sort font pairs by highest match count
void sortt(vector< pair<double, string> > &sorted);
struct Mats {
	Mat group; // Image stored
	string word; // Text detected on the image
	vector<float> confidence;	// Confidence of the text detected
	bool active = false;	// True if this struct is being processed by character recognition part

	string getWords() {
		return word;
	}
	float getAvgConfidence() {
		float ret = 0;
		for (auto f : confidence) ret += f;
		return ret / confidence.size();
	}
	float getHighestConf() {
		float ret = 0;
		for (auto f : confidence) if (f > ret) ret = f;
		return ret;
	}
};


//Perform text detection and recognition and evaluate results
int main(int argc, char* argv[])
{
	cout << endl << argv[0] << endl << endl;
	cout << "Program automaticaly recognizes font and returns best 5 matches " << endl;
	cout << "Dusan Javorek" << endl;

	cout << endl << "Please select mode:" << endl;
	cout << "1 - single image named `0.png` from Input folder" << endl;
	cout << "9 - run tests" << endl;
	char input;
	cin >> input;

	if (!initFontDatabase()) {
		cout << "ERROR: Failed to load font database";
		return -1;
	}
	if (input == '1') {
		FontRecognition();
	}
	if (input == '9') {

		int noFonts = getNfonts();

		for (int i = 0; i < noFonts; i++) {
			FontRecognition(i);
		}
		test(noFonts);
	}
	system("pause");
	return 0;
}
void FontRecognition(int testCase) {

	Mat image;
	if (testCase == -1) 
		image = imread("..\\Input\\0.png");
	else
		image = imread("../TestDatabase/Tests/" + to_string(testCase) + ".png");


	/*Text Detection*/

	// Extract channels to be processed individually
	vector<Mat> channels;

	Mat grey;
	cvtColor(image, grey, COLOR_RGB2GRAY);

	// Notice here we are only using grey channel, see textdetection.cpp for example with more channels
	channels.push_back(grey);
	channels.push_back(255 - grey);

	double t_d = (double)getTickCount();
	// Create ERFilter objects with the 1st and 2nd stage default classifiers
	Ptr<ERFilter> er_filter1 = createERFilterNM1(loadClassifierNM1("trained_classifierNM1.xml"), 8, 0.00015f, 0.13f, 0.2f, true, 0.1f);
	Ptr<ERFilter> er_filter2 = createERFilterNM2(loadClassifierNM2("trained_classifierNM2.xml"), 0.5);

	vector<vector<ERStat> > regions(channels.size());
	// Apply the default cascade classifier to each independent channel (could be done in parallel)
	for (int c = 0; c < (int)channels.size(); c++)
	{
		er_filter1->run(channels[c], regions[c]);
		er_filter2->run(channels[c], regions[c]);
	}
	cout << "TIME_REGION_DETECTION = " << ((double)getTickCount() - t_d) * 1000 / getTickFrequency() << endl;

	Mat out_img_decomposition = Mat::zeros(image.rows + 2, image.cols + 2, CV_8UC1);
	vector<Vec2i> tmp_group;
	for (int i = 0; i < (int)regions.size(); i++)
	{

		for (int j = 0; j < (int)regions[i].size(); j++)
		{
			tmp_group.push_back(Vec2i(i, j));
		}
		Mat tmp = Mat::zeros(image.rows + 2, image.cols + 2, CV_8UC1);
		er_draw(channels, regions, tmp_group, tmp);
		if (i > 0)
			tmp = tmp / 2;
		out_img_decomposition = out_img_decomposition | tmp;
		tmp_group.clear();
	}

	double t_g = (double)getTickCount();
	// Detect character groups
	vector< vector<Vec2i> > nm_region_groups;
	vector<Rect> nm_boxes;
	erGrouping(image, channels, regions, nm_region_groups, nm_boxes, ERGROUPING_ORIENTATION_HORIZ);
	cout << "TIME_GROUPING = " << ((double)getTickCount() - t_g) * 1000 / getTickFrequency() << endl;



	/*Text Recognition (OCR)*/

	double t_r = (double)getTickCount();
	Ptr<OCRTesseract> ocr = OCRTesseract::create();
	cout << "TIME_OCR_INITIALIZATION = " << ((double)getTickCount() - t_r) * 1000 / getTickFrequency() << endl;
	string output;

	Mat out_img;
	Mat out_img_detection;
	Mat out_img_segmentation = Mat::zeros(image.rows + 2, image.cols + 2, CV_8UC1);
	Mats *group_imgs = new Mats[(int)nm_boxes.size()];
	image.copyTo(out_img);
	image.copyTo(out_img_detection);
	float scale_img = 600.f / image.rows;
	float scale_font = (float)(2 - scale_img) / 1.4f;
	vector<string> words_detection;

	t_r = (double)getTickCount();


	for (int i = 0; i < (int)nm_boxes.size(); i++)
	{


		Mat group_img = Mat::zeros(image.rows + 2, image.cols + 2, CV_8UC1);
		er_draw(channels, regions, nm_region_groups[i], group_img);
		Mat group_segmentation;
		group_img.copyTo(group_segmentation);
		group_img(nm_boxes[i]).copyTo(group_img);
		group_img.copyTo(group_imgs[i].group);
		copyMakeBorder(group_img, group_img, 15, 15, 15, 15, BORDER_CONSTANT, Scalar(0));

		vector<Rect>   boxes;
		vector<string> words;
		vector<float>  confidences;
		ocr->run(group_img, output, &boxes, &words, &confidences, OCR_LEVEL_WORD);



		output.erase(remove(output.begin(), output.end(), '\n'), output.end());
		if (output.size() < 3)
			continue;


		for (int j = 0; j < (int)boxes.size(); j++)
		{
			boxes[j].x += nm_boxes[i].x - 15;
			boxes[j].y += nm_boxes[i].y - 15;

			if ((words[j].size() < 2) || (confidences[j] < 45) ||
				((words[j].size() == 2) && (words[j][0] == words[j][1])) ||
				((words[j].size() < 4) && (confidences[j] < 68)) ||
				isRepetitive(words[j]))
				continue;


			group_imgs[i].confidence.push_back(confidences[j]);
			group_imgs[i].word += words[j];
			group_imgs[i].active = true;

			words_detection.push_back(words[j]);



			out_img_segmentation = out_img_segmentation | group_segmentation;
		}

	}

	cout << "TIME_OCR = " << ((double)getTickCount() - t_r) * 1000 / getTickFrequency() << endl;


	/* Character segmentation*/

	double t_s = (double)getTickCount();
	int name_index = 0;
	vector<Mats> words;
	double fixed_height = 300;
	bool segmented = false;
	
	for (int i = 0; i < (int)nm_boxes.size(); i++) {
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		vector<Rect> rs;


		// skip all uncertain chunks
		// words with confidence higher then 68% will be considered
		if (group_imgs[i].getHighestConf() < 68 ||
			group_imgs[i].getAvgConfidence() < 60 ||
			!group_imgs[i].active) continue;

		
		// Resize and apply threshold
		double width = group_imgs[i].group.cols * (350.0 / group_imgs[i].group.rows);
		resize(group_imgs[i].group, group_imgs[i].group, Size(width, 350), 0, 0, INTER_CUBIC);
		threshold(group_imgs[i].group, group_imgs[i].group, 50, 255, THRESH_BINARY);

		findContours(group_imgs[i].group, contours, hierarchy,
			CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

		// Find edges on gray scale image
		// Adapt thresholds based on Otsu’s algorithm
		// blur the image by kernel 5x5

		if (nm_boxes[i].br().x >= image.cols) nm_boxes[i].width -= nm_boxes[i].br().x - image.cols + 1;
		if (nm_boxes[i].br().y >= image.rows) nm_boxes[i].height -= nm_boxes[i].br().y - image.rows + 1;
		Mat canny = Mat(image, nm_boxes[i]);

		Mat otsu;
		double high_thresh, thresh;

		width = canny.cols * (350.0 / canny.rows);
		cvtColor(canny, canny, CV_BGR2GRAY);
		resize(canny, canny, Size(width, 350), 0, 0, INTER_CUBIC);
		blur(canny, canny, Size(5, 5));
		double threshold_ = threshold(canny, otsu, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		high_thresh = threshold_;
		thresh = high_thresh*0.8;

		//imwrite("../output/DEBUG1" + group_imgs[i].word + to_string(i) + to_string(testCase) + ".png", canny);
		Canny(canny, canny, thresh, high_thresh, 3);
		//imwrite("../output/DEBUG0" + group_imgs[i].word + to_string(i) + to_string(testCase) + ".png", canny);


		int index = 0;
		vector<Mat> temp;
		// iterate through all the top-level contours
		// and divide them each to separate matrix
		// Also check average height for false contours
		for (vector<Point> c : contours) {


			Rect r = boundingRect(c);
			int margin = 7;
			if (!(r.x - margin < 0)) { r.x -= margin; r.width += margin; }
			if (!(r.y - margin < 0)) { r.y -= margin; r.height += margin; }

			if (r.br().x >= canny.cols) r.width -= r.br().x - canny.cols + 1;
			if (r.br().y >= canny.rows) r.height -= r.br().y - canny.rows + 1;
			
			// We want only top level contours -> only letters
			if (hierarchy[index][3] >= 0) {
				index++;
				continue;
			}


			temp.push_back(Mat(canny, r));

			rs.push_back(r);

			index++;


		}

		sortt(rs, temp);

		for (int j = 0; j < rs.size(); j++) {

			// Assign letters
			Mats word;
			if (j < group_imgs[i].word.size())
				word.word = group_imgs[i].word[j];
			else
				word.word = to_string(name_index);
			// If letter was `i` we want the dot also
			// Can be adjusted for all letters with interpuction
			if (word.word == "i") { rs[j].y = 0; rs[j].height = canny.rows; }
			Mat tt = Mat(canny, rs[j]);
			tt.copyTo(word.group);



			// Make border around for better SURF detection and resize picture
			copyMakeBorder(word.group, word.group, 10, 10, 10, 10, BORDER_CONSTANT, Scalar());
			width = word.group.cols * (fixed_height / word.group.rows);
			resize(word.group, word.group, Size(width, 300), 0, 0, INTER_CUBIC);


			//imwrite("../Output/" + to_string(j) + ".jpg", word.group);
			words.push_back(word);

			name_index++;
		}
	}

	cout << "TIME_WORD_SEGMENTATION = " << ((double)getTickCount() - t_s) * 1000 / getTickFrequency() << endl;
	
	t_s = (double)getTickCount();
	
	/* Font detection */
	double t_f = (double)getTickCount();


	// with minHessian = 200 SURF should produce at least 400 keypoints for best preformance
	int minHessian = 200;
	vector< vector <FontDescriptor> > fontDescriptors(255);

	

	// Init font pairs
	vector< pair <double, string> > fonts;
	int noFonts = getNfonts();
	for (int i = 0; i < getNfonts(); i++) {
		fonts.push_back(pair<int, string>(-1, getFonts()[i]));
	}


	Ptr<SURF> detector = SURF::create(minHessian, 1, 3, true, false);

	for (int i = 0; i < words.size(); i++) {

		// Number of sectors image will be divided into
		int noSectors = 6;
		// Keypoints for letters
		vector <KeyPoint> k1;
		vector < vector < KeyPoint > > _k1(noSectors);
		// Descriptors
		vector < Mat > d1(noSectors);

		int wordIndex = (int)words[i].word[0];
		vector < Rect > sectors;
		// Corners of images
		Point2f corner1(0, 0), corner10(500, 500);
		// Load font descriptors for every font of one specific letter
		// If we havent already loaded requested letter, load it

		if (fontDescriptors[wordIndex].size() == 0)
			fontDescriptors[wordIndex] = getDescriptors(words[i].word[0]);

		// Detect and compute keypoints on the edges of letter using SURF
		detector->detect(words[i].group, k1);

		// Match current descriptor with font descriptors and save best results

		// Brute-Force matcher
		BFMatcher BF(NORM_L2, true);

		// Get close rectangle around letter
		for (int j = 0; j < k1.size(); j++) {
			if (k1[j].pt.x > corner1.x) corner1.x = k1[j].pt.x;
			if (k1[j].pt.y > corner1.y) corner1.y = k1[j].pt.y;

			if (k1[j].pt.x < corner10.x) corner10.x = k1[j].pt.x;
			if (k1[j].pt.y < corner10.y) corner10.y = k1[j].pt.y;
		}
		corner1.x += 10;
		corner1.y += 10;
		corner10.x -= 10;
		corner10.y -= 10;

		// Divide image into small sectors
		int secWidth, secHeight;
		secWidth = (corner1.x - corner10.x) / 2; secHeight = (corner1.y - corner10.y) / 3;

		sectors.push_back(Rect((int)corner10.x, (int)corner10.y, secWidth, secHeight));
		sectors.push_back(Rect((int)corner10.x, corner10.y + secHeight, secWidth, secHeight));
		sectors.push_back(Rect((int)corner10.x, corner10.y + secHeight * 2, secWidth, secHeight));

		sectors.push_back(Rect((int)corner10.x + secWidth, (int)corner10.y, secWidth, secHeight));
		sectors.push_back(Rect((int)corner10.x + secWidth, corner10.y + secHeight, secWidth, secHeight));
		sectors.push_back(Rect((int)corner10.x + secWidth, corner10.y + secHeight * 2, secWidth, secHeight));
		
		// Divide keypoints into sectors
		for (int j = 0; j < noSectors; j++) {

			for (int k = 0; k < k1.size(); k++) {
				if (k1[k].pt.x >= sectors[j].x && k1[k].pt.x <= sectors[j].br().x &&
					k1[k].pt.y >= sectors[j].y && k1[k].pt.y <= sectors[j].br().y) {
					_k1[j].push_back(k1[k]);
				}
			}
		}

		// Compute descriptors
		for (int j = 0; j < noSectors; j++) {
			detector->compute(words[i].group, _k1[j], d1[j]);
		}


		for (int f = 0; f < noFonts; f++) {
			int noInliners = -1;
			// Mat for displaying results
			Mat drawMatchesImg;

			// Keypoints and descriptors for word from database

			vector <vector< KeyPoint > > _k2 = fontDescriptors[wordIndex][f].section_keypoints;
			vector < KeyPoint > k2 = fontDescriptors[wordIndex][f].keypoints;
			vector< Mat > d2 = fontDescriptors[wordIndex][f].descriptor;
			
			// Matches
			vector <vector < DMatch> > matches(noSectors);
			vector <DMatch> goodMatches;


			// Matching
			for (int j = 0; j < noSectors; j++) {
				if (d2[j].rows && d1[j].rows)
					BF.match(d2[j], d1[j], matches[j]);
			}

			
			// Unite keypoints and matches back into 1 vector
			k1.clear();

			for (int k = 0; k < noSectors; k++) {

				for (int j = 0; j < _k1[k].size(); j++)
					k1.push_back(_k1[k][j]);
				for (int j = 0; j < matches[k].size(); j++) {
					if (k > 0) {

						// Rename indexes
						int offset1 = 0, offset2 = 0;
						for (int l = 0; l < k; l++)
							offset1 += _k1[l].size();
						for (int l = 0; l < k; l++)
							offset2 += _k2[l].size();

						matches[k][j].queryIdx += offset2;
						matches[k][j].trainIdx += offset1;
					}
				}

				// Add only good matches

				// Recycling
				double max_dist = 0; double min_dist = 100;

				for (int j = 0; j < matches[k].size(); j++)
				{
					double dist = matches[k][j].distance;
					if (dist < min_dist) min_dist = dist;
					if (dist > max_dist) max_dist = dist;
				}

				for (int j = 0; j < matches[k].size(); j++) {
					if (matches[k][j].distance < 5.0 * min_dist)
						goodMatches.push_back(matches[k][j]);
				}
			}

			/*drawMatches(fontDescriptors[wordIndex][f].img, k2,
				words[i].group, k1,
				goodMatches, drawMatchesImg,
				Scalar(255, 50, 50), Scalar(50, 50, 255), vector<char>(), DrawMatchesFlags::DEFAULT);*/
			/*drawMatches(fontDescriptors[wordIndex][f].img, vector<KeyPoint>(),
				words[i].group, vector<KeyPoint>(),
				vector<DMatch>(), drawMatchesImg,
				Scalar(255, 50, 50), Scalar(50, 50, 255), vector<char>(), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);*/
			
			// Homography
			vector<Point2f> srcPts;
			vector<Point2f> dstPts;
			double letter_ratio = 0;

			// We need enough matches to compute homography
			if (goodMatches.size() > 10) {
				for (int j = 0; j < goodMatches.size(); j++) {
					srcPts.push_back(k2[goodMatches[j].queryIdx].pt);

					dstPts.push_back(k1[goodMatches[j].trainIdx].pt);
				}
				Mat mask;
				Mat M = findHomography(srcPts, dstPts, RANSAC, 5.0, mask);


				vector < Point2f > goodScrPoints, goodDstPoints;

				for (int j = 0; j < mask.rows; j++) {
					// We want only matches that define homography
					if ((unsigned int)mask.at<uchar>(j)) {
						goodScrPoints.push_back(srcPts[j]);
						goodDstPoints.push_back(dstPts[j]);

						/*circle(drawMatchesImg, srcPts[j], 5, Scalar(255, 0, 0), 1);
						circle(drawMatchesImg, dstPts[j] + Point2f(fontDescriptors[wordIndex][f].img.cols, 0), 5, Scalar(255, 0, 0), 1);
						line(drawMatchesImg, srcPts[j], dstPts[j] + Point2f(fontDescriptors[wordIndex][f].img.cols, 0), Scalar(0, 0, 255), 1);*/
					}
				}
				// No. of best matched keypoints
				noInliners = goodScrPoints.size();

				vector <Point2f> scenePts(4), objPts(4);
				Point2f scPts1(5000, 5000), scPts2(0, 0);
				/*scenePts[0] = corner10; scenePts[1] = Point2f(corner1.x, corner10.y);
				scenePts[2] = corner1;  scenePts[3] = Point2f(corner10.x, corner1.y);*/
				for (int p = 0; p < k2.size(); p++) {
					if (k2[p].pt.x < scPts1.x) scPts1.x = k2[p].pt.x;
					if (k2[p].pt.y < scPts1.y) scPts1.y = k2[p].pt.y;

					if (k2[p].pt.x > scPts2.x) scPts2.x = k2[p].pt.x;
					if (k2[p].pt.y > scPts2.y) scPts2.y = k2[p].pt.y;
				}
				scPts2.x += 10;
				scPts2.y += 10;
				scPts1.x -= 10;
				scPts1.y -= 10;
				scenePts[0] = scPts1; scenePts[1] = Point2f(scPts2.x, scPts1.y);
				scenePts[2] = scPts2;  scenePts[3] = Point2f(scPts1.x, scPts2.y);

				perspectiveTransform(scenePts, objPts, M);

				// Ratio of diagonals between transformed rectangle
				/*letter_ratio = sqrt(pow(objPts[0].x - objPts[2].x, 2) + pow(objPts[0].y - objPts[2].y, 2)) /
							   sqrt(pow(objPts[1].x - objPts[3].x, 2) + pow(objPts[1].y - objPts[3].y, 2));
				line(drawMatchesImg, objPts[0] + Point2f(words[i].group.cols, 0), objPts[1] + Point2f(words[i].group.cols, 0), Scalar(0, 255, 0), 2);
				line(drawMatchesImg, objPts[1] + Point2f(words[i].group.cols, 0), objPts[2] + Point2f(words[i].group.cols, 0), Scalar(0, 255, 0), 2);
				line(drawMatchesImg, objPts[2] + Point2f(words[i].group.cols, 0), objPts[3] + Point2f(words[i].group.cols, 0), Scalar(0, 255, 0), 2);
				line(drawMatchesImg, objPts[3] + Point2f(words[i].group.cols, 0), objPts[0] + Point2f(words[i].group.cols, 0), Scalar(0, 255, 0), 2);

				line(drawMatchesImg, scenePts[0], scenePts[1], Scalar(0, 255, 0), 2);
				line(drawMatchesImg, scenePts[1], scenePts[2], Scalar(0, 255, 0), 2);
				line(drawMatchesImg, scenePts[2], scenePts[3], Scalar(0, 255, 0), 2);
				line(drawMatchesImg, scenePts[3], scenePts[0], Scalar(0, 255, 0), 2);*/

				//imwrite("../Output/" + words[i].word + to_string(i) + to_string(f) + ".png", drawMatchesImg);
			} else
				noInliners++;
			fonts[f].first += noInliners;
			//fonts[f].first += letter_ratio;
		}
	}
	/*for (int i = 0; i < noFonts; i++) {
		fonts[i].first /= words.size();
		fonts[i].first = abs(1 - fonts[i].first);
		cout << fonts[i].second << " " << fonts[i].first << endl;
	}*/
	// Sort font by best match value
	sortt(fonts);

	int TOP_FONTS_DISPLAY = 5;
	if (noFonts < TOP_FONTS_DISPLAY) TOP_FONTS_DISPLAY = noFonts;
	vector< pair <int, string> > TOPfonts;
	for (int i = 0; i < TOP_FONTS_DISPLAY; i++) {
		TOPfonts.push_back(fonts[i]);
	}
	cout << "TIME_FONT_DETECTION = " << ((double)getTickCount() - t_f) * 1000 / getTickFrequency() << endl;

	// Display results
	displayFonts(TOPfonts, testCase);



}

size_t min(size_t x, size_t y, size_t z)
{
	return x < y ? min(x, z) : min(y, z);
}

bool isRepetitive(const string& s)
{
	int count = 0;
	for (int i = 0; i<(int)s.size(); i++)
	{
		if ((s[i] == 'i') ||
			(s[i] == 'l') ||
			(s[i] == 'I'))
			count++;
	}
	if (count >((int)s.size() + 1) / 2)
	{
		return true;
	}
	return false;
}


void er_draw(vector<Mat> &channels, vector<vector<ERStat> > &regions, vector<Vec2i> group, Mat& segmentation)
{
	for (int r = 0; r<(int)group.size(); r++)
	{
		ERStat er = regions[group[r][0]][group[r][1]];
		if (er.parent != NULL) // deprecate the root region
		{
			int newMaskVal = 255;
			int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
			floodFill(channels[group[r][0]], segmentation, Point(er.pixel%channels[group[r][0]].cols, er.pixel / channels[group[r][0]].cols),
				Scalar(255), 0, Scalar(er.level), Scalar(0), flags);
		}
	}
}

bool   sort_by_lenght(const string &a, const string &b) { return (a.size()>b.size()); }


void sortt(vector<Rect> &rs, vector<Mat> &gr) {

	int _size = rs.size();
	if (_size == 0) return;

	struct Comp {
		Rect r;
		Mat m;
	};
	Comp *comp = new Comp[_size];

	for (int i = 0; i < _size; i++) {
		comp[i].r = rs[i];
		comp[i].m = gr[i];
	}

	qsort(comp, _size, sizeof(Comp), [](const void *a, const void *b)->int {
		if ( ((Comp*)a)->r.x < ((Comp*)b)->r.x ) return -1;
		if (((Comp*)a)->r.x == ((Comp*)b)->r.x) return 0;
		if (((Comp*)a)->r.x > ((Comp*)b)->r.x) return 1;
	});

	rs.clear(); gr.clear();

	for (int i = 0; i < _size; i++) {
		rs.push_back(comp[i].r);
		gr.push_back(comp[i].m);
	}
}
void sortt(vector< pair<double, string> > &sorted) {

	int _size = sorted.size();
	if (_size == 0) return;

	pair<double, string> *s = new pair<double, string>[_size];
	for (int i = 0; i < _size; i++)
		s[i] = sorted[i];
	
	qsort(s, _size, sizeof(pair<double, string>), [](const void *a, const void *b)->int {
		if (((pair<double, string>*)a)->first <  ((pair<double, string>*)b)->first) return 1;
		if (((pair<double, string>*)a)->first == ((pair<double, string>*)b)->first) return 0;
		if (((pair<double, string>*)a)->first >  ((pair<double, string>*)b)->first) return -1;
	});

	sorted.clear();

	for (int i = 0; i < _size; i++) {
		sorted.push_back(s[i]);
	}
}
