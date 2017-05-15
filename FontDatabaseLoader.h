#ifndef _FONTDATATABELOADER_
#define __FONTDATATABELOADER__

#include "opencv2/core/utility.hpp"

using namespace std;
using namespace cv;

const string DATABASE = "..\\..\\Database";

// Delete keypoints and img - testing purpose only

struct FontDescriptor {
	vector < Mat > descriptor;
	vector < vector<KeyPoint> > section_keypoints;
	vector < KeyPoint> keypoints;
	Mat img;

	string font;
	FontDescriptor(string _font, vector <Mat> _descriptor, vector <vector<KeyPoint>> _section_keypoints, vector < KeyPoint> _keypoints, Mat _img) { 
		font = _font; 
		descriptor = _descriptor;
		section_keypoints = _section_keypoints;
		keypoints = _keypoints; 
		img = _img; 
	}
};

vector<FontDescriptor> getDescriptors(char letter);
int initFontDatabase();
int getNfonts();
vector<string> getFonts();
void displayFonts(vector < pair <int, string> > &TOPfonts, int testCase = -1);


#endif // !FONTDATATABELOADER__H
