#include "FontDatabaseLoader.h"
#include "DataStructures.h"

#include "opencv2/highgui.hpp"

#include <iostream>
#include <fstream>


vector<string> fonts;

int loadFonts() {
	ifstream list(DATABASE + "\\_Config\\list.txt");

	if (!list.is_open()) return 0;

	string font;
	getline(list, font);
	while (!list.eof()) {
		fonts.push_back(font);
		getline(list, font);
	}
	list.close();
	return fonts.size();
}
vector<FontDescriptor> getDescriptors(char letter) {

	vector<FontDescriptor> ret;

	for (int i = 0; i < fonts.size(); i++) {

		FileStorage data(DATABASE + "\\" + fonts[i] + "\\" + to_string((int)letter) + ".xml", FileStorage::READ);
		
		vector < Mat > desc;
		vector < vector<KeyPoint> >  section_keypoints;
		vector < KeyPoint > keypoints;
		int noSectors = 0;
		data["sectors"] >> noSectors;
		for (int j = 0; j < noSectors; j++) {
			
			Mat _desc;
			vector <KeyPoint> _k;
			data["descriptor" + to_string(j)] >> _desc;
			
			data["keypoints" + to_string(j)] >> _k;

			desc.push_back(_desc);
			section_keypoints.push_back(_k);

		}
		data["keypoints"] >> keypoints;
		Mat img = imread(DATABASE + "\\" + fonts[i] + "\\" + to_string((int)letter) + ".png");
		ret.push_back(FontDescriptor(fonts[i], desc, section_keypoints, keypoints, img));
		
		data.release();
	}
	return ret;
}
int initFontDatabase() {
	return loadFonts();
}
int getNfonts() {
	return fonts.size();
}
vector<string> getFonts() {
	return fonts;
}
void displayFonts(vector < pair <int, string> > &TOPfonts, int testCase) {
	if (TOPfonts[0].first == -1) {
		cout << "ERROR: Segmentation or OCR fail" << endl;
	} else 
	if (TOPfonts[0].first == 0)  {
		cout << "ERROR: Not enough keypoints" << endl;
	}
	else {
		cout << "--------------------------------" << endl;
		cout << "    TOP " << TOPfonts.size() << " fonts matched:" << endl;
		for (int i = 0; i < TOPfonts.size(); i++) {
			cout << TOPfonts[i].second << endl;
		}
		cout << "--------------------------------" << endl;
	}

	if (testCase > -1) {
		ofstream f("..\\Output\\" + to_string(testCase) + ".txt");
		if (TOPfonts[0].first == -1)
			f << SEGMENTATION_OCR_FAIL << endl;
		else
		if (TOPfonts[0].first == 0)
			f << NOT_ENOUGH_KEYPOINTS_FAIL << endl;
		else
			for (int i = 0; i < TOPfonts.size(); i++) {
				f << TOPfonts[i].second << endl;
			}
		f.close();
	}
}