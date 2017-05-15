#include <fstream>
#include <iostream>
#include <iomanip>

#include "opencv2/core/utility.hpp"

#include "DataStructures.h"

using namespace std;
using namespace cv;
/* Tests the result of program, counts error rate during font recognition and error rate before font recognition (mostly ocr)
   Outputs the successful rate on what position the program calculated the right font - 
   Top 1. result means how many font were calculated right as the first guess. 
   Top 5. result means how many fonts were calculated right in the whole output / per image as program outputs top 5 fonts for each image.
*/
void test(int noTests) {
	float result;
	int score[5];
	int error = 0;
	int e_error = 0;
	for (int i = 0; i < 5; i++)
		score[i] = 0;
	for (int i = 0; i < noTests; i++) {
		ifstream database("..\\TestDatabase\\Tests\\" + to_string(i) + ".txt");
		ifstream program("..\\Output\\" + to_string(i) + ".txt");

		string d, p;
		getline(database, d);
		for (int j = 0; j < 5; j++) {
			if (program.eof()) break;
			getline(program, p);
			
			if (j == 0) {
				if (!p.compare(SEGMENTATION_OCR_FAIL)) {
					e_error++;
				}
				if (!p.compare(NOT_ENOUGH_KEYPOINTS_FAIL)) {
					error++;
				}
			}
			if (!d.compare(p)) {
				for (int k = j; k < 5; k++)
					score[k] ++;
				break;
			}
		}
		database.close();
		program.close();
	}
	setprecision(3);
	cout << "Recognized font texts: " << (noTests - error - e_error) << "/" << noTests << endl;
	cout << "Successful rate on recognized fonts: " << endl;
	for (int i = 0; i < 5; i++) {
		result = score[i] / ((noTests - error - e_error) * 1.0);
		cout << "Top " << i + 1 << ". result: " << setw(3) << result * 100.0 << "%" << endl;
	}
	cout << "Error rate: " << setw(3) << error / (noTests*1.0) * 100 << "%" << endl;
	cout << "External error rate: " << setw(3) << e_error / (noTests*1.0) * 100 << "%" << endl;
}