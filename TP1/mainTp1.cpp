#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

using namespace cv;
using namespace std;

//=======================================================================================
// to YCrCb
//=======================================================================================
void toYCrCb(const Mat & imgSrc, Mat & imgDestY, Mat & imgDestCr, Mat & imgDestCb){
	Mat imgYCrCb;
	vector<Mat> img3cans;
	cvtColor(imgSrc, imgYCrCb, CV_BGR2YCrCb);
	split(imgYCrCb, img3cans);
	imgDestY = img3cans[0];
	imgDestCr = img3cans[1];
	imgDestCb = img3cans[2];

	// imshow("InputImageSrcY", imgDestY);
	// cvWaitKey();
	// imshow("InputImageSrcCr", imgDestCr);
	// cvWaitKey();
	// imshow("InputImageSrcCb", imgDestCb);
  // cvWaitKey();
}

//=======================================================================================
// computeHistogram
//=======================================================================================
void computeHistogram(const Mat& inputComponent, Mat& myHist)
{
	/// Establish the number of bins
	int histSize = 256;
	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 } ;
	const float* histRange = { range };
	bool uniform = true;
	bool accumulate = false;

	/// Compute the histograms:
	calcHist( &inputComponent, 1, 0, Mat(), myHist, 1, &histSize, &histRange, uniform, accumulate );
}

//=======================================================================================
// displayHistogram
//=======================================================================================
void displayHistogram(const Mat& myHist)
{
	// Establish the number of bins
	int histSize = 256;
	// Draw one histogram
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound( (double) hist_w/histSize );
	Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
	/// Normalize the result to [ 0, histImage.rows ]
	Mat myHistNorm;
	normalize(myHist, myHistNorm, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

	/// Draw for each channel
	for( int i = 1; i < histSize; i++ )
	{
		line( histImage, Point( bin_w*(i-1), hist_h - cvRound(myHistNorm.at<float>(i-1)) ) , Point( bin_w*(i), hist_h - cvRound(myHistNorm.at<float>(i)) ), Scalar( 255, 255, 255), 2, 8, 0 );
	}
	/// Display
	namedWindow("Display Histo", CV_WINDOW_AUTOSIZE );
	imshow("Display Histo", histImage );
	cvWaitKey();
}

//=======================================================================================
// Mat norm_0_255(InputArray _src)
// Create and return normalized image
//=======================================================================================
Mat norm_0_255(InputArray _src) {
 Mat src = _src.getMat();
 // Create and return normalized image:
 Mat dst;
 switch(src.channels()) {
	case 1:
		normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
	src.copyTo(dst);
	break;
 }
 return dst;
}

//=======================================================================================
// EQM
//=======================================================================================
double eqm(const Mat & img1, const Mat & img2)
{
	int rows = img1.rows;
	int cols = img1.cols;

	if(rows != img2.rows || cols != img2.cols){
		cout << "Error, images need to have the same size. \n";
	}

	double EQM;

	for(int i = 0; i < rows; i++){
		for(int j = 0; j < cols; j++){
			EQM += pow(img1.at<unsigned char>(i, j) - img2.at<unsigned char>(i,j), 2);
		}
	}

	EQM /= (rows * cols);
	cout << "EQM : " << EQM << "\n";
 	return EQM;
}

//=======================================================================================
// psnr
//=======================================================================================
double psnr(const Mat & imgSrc, const Mat & imgDeg)
{

	double EQM = eqm(imgSrc, imgDeg);

	if(EQM == 0){
		cout << "Error, PSNR can't compare two same images \n";
		return -1;
	}
	double psnr = 10 * log10(65025/EQM);
	cout << "PSNR : " << psnr << "\n";
 	return psnr;
}

//=======================================================================================
// distortionMap
//=======================================================================================
void distortionMap(const vector<Mat> & imgSrc, const vector<Mat> & imgDeg, Mat &distoMap)
{
	vector<Mat> tmp;
	int alpha = 1;

	tmp[0] = imgSrc[0] - imgDeg[0];
	tmp[1] = imgSrc[1] - imgDeg[1];
	tmp[2] = imgSrc[2] - imgDeg[2];

	tmp[0] = 128 * (alpha * tmp[0]);
	tmp[1] = 128 * (alpha * tmp[1]);
	tmp[2] = 128 * (alpha * tmp[2]);

	cvMerge(tmp, distoMap);

}

//=======================================================================================
//=======================================================================================
// MAIN
//=======================================================================================
//=======================================================================================
int main(int argc, char** argv){
  if (argc < 3){
    std::cout << "No image data or not enough ... At least two argument is required! \n";
    return -1;
  }

  Mat inputImageSrc;
	Mat inputImageSrc2;
	Mat imgSrcY, imgSrcCr, imgSrcCb;
	Mat imgDegY, imgDegCr, imgDegCb;

  // Ouvrir l'image d'entr�e et v�rifier que l'ouverture du fichier se d�roule normalement
  inputImageSrc = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	inputImageSrc2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
  if(!inputImageSrc.data ) { // Check for invalid input
        std::cout <<  "Could not open or find the image " << argv[1] << std::endl ;
		waitKey(0); // Wait for a keystroke in the window
        return -1;
  }
	if(!inputImageSrc2.data ) { // Check for invalid input
        std::cout <<  "Could not open or find the image " << argv[2] << std::endl ;
		waitKey(0); // Wait for a keystroke in the window
        return -1;
  }

	//Conversion en YCrCb
	toYCrCb(inputImageSrc, imgSrcY, imgSrcCr, imgSrcCb);
	toYCrCb(inputImageSrc2, imgDegY, imgDegCr, imgDegCb);

  // Visualiser l'image
	imshow("InputImageSrcBGR", inputImageSrc);
	cvWaitKey();
	//Enregistrement des inputImageSrc
	imwrite((string)argv[1] + "_Y", imgSrcY);
	imwrite((string)argv[1] + "_Cb", imgSrcCb);
	imwrite((string)argv[1] + "_Cr", imgSrcCr);
	// Visualiser l'image
	imshow("InputImageSrc2BGR", inputImageSrc2);
	cvWaitKey();
	//Enregistrement des inputImageSrc
	imwrite((string)argv[2] + "_Y", imgDegY);
	imwrite((string)argv[2] + "_Cb", imgDegCb);
	imwrite((string)argv[2] + "_Cr", imgDegCr);

	cout << "Calculs between " << argv[1] << "_Y and " << argv[2] << "_Y : "<< "\n";
	psnr(imgSrcY, imgDegY);
	cout << "Calculs between " << argv[1] << "_Cr and " << argv[2] << "_Cr : "<< "\n";
	psnr(imgSrcCr, imgDegCr);
	cout << "Calculs between " << argv[1] << "_Cb and " << argv[2] << "_Cb : "<< "\n";
	psnr(imgSrcCb, imgDegCb);

	vector<Mat> imgSrc;
	cvMerge(imgSrcY, imgSrcCr, imgSrcCb, imgSrc);
	vector<Mat> imgDeg;
	cvMerge(imgDegY, imgDegCr, imgDegCb, imgDeg);

	Mat errorMap;
	distortionMap(imgSrc, imgDeb, errorMap);

	imshow(errorMap);

  return 0;
}
