//Disclaimer: a lot of this code was taken and adapted by combing through the openCV docs and examples

#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat frame);
bool compare_rect(const Rect & a, const Rect &b);


/** Global variables */
String face_cascade_name, left_eyes_cascade_name, right_eyes_cascade_name;
CascadeClassifier face_cascade;
CascadeClassifier left_eyes_cascade;
CascadeClassifier right_eyes_cascade;
String window_name = "Capture - Face detection";
int eyesState = -1;
bool drawEyes = true;

/** @function main */
int main()
{
	face_cascade_name = "C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml";
	left_eyes_cascade_name = "C:\\opencv\\sources\\data\\haarcascades\\haarcascade_lefteye_2splits.xml";
	right_eyes_cascade_name = "C:\\opencv\\sources\\data\\haarcascades\\haarcascade_righteye_2splits.xml";
	VideoCapture capture;
	Mat frame;

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading face cascade\n"); return -1; };
	if (!left_eyes_cascade.load(left_eyes_cascade_name)) { printf("--(!)Error loading eyes cascade\n"); return -1; };
	if (!right_eyes_cascade.load(right_eyes_cascade_name)) { printf("--(!)Error loading eyes cascade\n"); return -1; };

	//-- 2. Read the video stream
	capture.open(0);
	if (!capture.isOpened()) { printf("--(!)Error opening video capture\n"); return -1; }

	while (capture.read(frame))
	{
		if (frame.empty())
		{
			printf(" --(!) No captured frame -- Break!");
			break;
		}

		//-- 3. Apply the classifier to the frame
		detectAndDisplay(frame);

		char c = (char)waitKey(10);
		if (c == 27) { break; } // escape
	}
	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay(Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	imshow("Greyscale", frame_gray);

	equalizeHist(frame_gray, frame_gray);
	imshow("Histogram Equalized", frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	//-- Sort faces with custom sort function
	sort(faces.begin(), faces.end(), compare_rect);
	if (faces.size() > 0)
	{
		int i = 0;

		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);

		//split face rectangle in half
		Rect faceRight = Rect(faces[i].x, faces[i].y, faces[i].width / 2, faces[i].height * 2 / 3);
		Rect faceLeft = Rect(faces[i].x + (faces[i].width / 2), faces[i].y, faces[i].width / 2, faces[i].height * 2 / 3);

		Mat faceROI_left = frame_gray(faceLeft);
		imshow("Left Eye Region of Interest", faceROI_left);

		Mat faceROI_right = frame_gray(faceRight);
		imshow("Right Eye Region of Interest", faceROI_right);

		std::vector<Rect> left_eyes;
		std::vector<Rect> right_eyes;

		//-- In each face, detect eyes
		left_eyes_cascade.detectMultiScale(faceROI_left, left_eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
		right_eyes_cascade.detectMultiScale(faceROI_right, right_eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

		int oldEyesState = eyesState;
		eyesState = ((left_eyes.size() == 1) && (right_eyes.size() == 1));


		if (oldEyesState != eyesState)
		{
			if (eyesState == 0)
			{
				//cout << "asleep" << endl;
				drawEyes = false;
			}
			else
			{
				//cout << "awake" << endl;
				drawEyes = true;
			}
		}

		for (size_t j = 0; j < right_eyes.size(); j++)
		{
			Point eye_center(faces[i].x + right_eyes[j].x + right_eyes[j].width / 2, faces[i].y + right_eyes[j].y + right_eyes[j].height / 2);
			int radius = cvRound((right_eyes[j].width + right_eyes[j].height)*0.25);
			if(drawEyes)
				circle(frame, eye_center, radius, Scalar(255, 0, 0), 4, 8, 0);
		}
		for (size_t j = 0; j < left_eyes.size(); j++)
		{
			Point eye_center(faces[i].x + faces[i].width / 2 + left_eyes[j].x + left_eyes[j].width / 2, faces[i].y + left_eyes[j].y + left_eyes[j].height / 2);
			int radius = cvRound((left_eyes[j].width + left_eyes[j].height)*0.25);
			if(drawEyes)
				circle(frame, eye_center, radius, Scalar(255, 0, 0), 4, 8, 0);
		}

		
	}
	//-- Show what you got
	imshow(window_name, frame);

	//screenshot option
	/*
	if (faces.size() > 0)
		waitKey();
		*/
}

/** @function compareRect */
bool compare_rect(const Rect & a, const Rect &b) 
{
	return a.width > b.width;
}
