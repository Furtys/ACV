#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

using namespace cv;
using namespace std;


//==================================================================================  a =====
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
// to YCrCb
//=======================================================================================
void toYCrCb(const Mat & imgSrc, Mat &ImgSrcYCrCb){
	cvtColor(imgSrc, ImgSrcYCrCb, CV_BGR2YCrCb);
}

//=======================================================================================
// computeHistogram
//=======================================================================================
void computeHistogram(const Mat& inputComponent, Mat& myHist)
{
  Mat absInputComponent = abs(inputComponent);
	/// Establish the number of bins
	int histSize = 256;
	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 } ;
	const float* histRange = { range };
	bool uniform = true;
	bool accumulate = false;

	/// Compute the histograms:
	calcHist( &absInputComponent, 1, 0, Mat(), myHist, 1, &histSize, &histRange, uniform, accumulate );
}

//=======================================================================================
// displayHistogram
//=======================================================================================
Mat displayHistogram(const Mat& myHist)
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

  return histImage;
}

//=======================================================================================
// Entropy
//=======================================================================================

void entropyCalculus(const Mat& errorMap, Mat& histo)
{
	float entropy = 0;
	int n = errorMap.rows * errorMap.cols;
	float proba = 0;
	for (int i = 0; i < histo.rows; ++i)
	{
		if (histo.at<float>(i) != 0)
		{
			proba = histo.at<float>(i)/n;
			entropy += - proba * log2(proba);
		}

	}
	std::cout << "ENTROPY : " << entropy << std::endl;
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
      double pixel1 = (double)img1.at<unsigned char>(i, j);
      double pixel2 = (double)img2.at<unsigned char>(i, j);
			EQM += (pixel1 - pixel2) * (pixel1 - pixel2);
		}
	}

	EQM /= (rows * cols);
	cout << "EQM : " << EQM << "\n";
 	return EQM;
}

//=======================================================================================
// psnr
//=======================================================================================
double psnr(const Mat & imgSrc, const Mat & imgDCT)
{
  Mat imgDCT8Usrc(imgSrc.size(), CV_8UC1);
  Mat imgDCT8UDCT(imgDCT.size(), CV_8UC1);
  imgSrc.convertTo(imgDCT8Usrc, CV_8UC1);
  imgDCT.convertTo(imgDCT8UDCT, CV_8UC1);

	double EQM = eqm(imgDCT8Usrc, imgDCT8UDCT);

	if(EQM == 0){
		cout << "Error, PSNR can't compare two identic images \n";
		return -1;
	}
	double psnr = 10 * log10(65025/EQM);
	cout << "PSNR : " << psnr << "\n";
 	return psnr;
}

//=======================================================================================
// DCT
//=======================================================================================
void DCT(const vector<Mat> &vecImgSrc, vector<Mat> &vecImgSrcDCT){
  for(int i = 0; i < 3; i++){
    Mat temp;
    dct(vecImgSrc[i], temp);
    vecImgSrcDCT.push_back(temp);
  }
}

//=======================================================================================
// DCT Inverse
//=======================================================================================
void inverseDCT(const vector<Mat> &vecImgSrcDCT, vector<Mat> &vecImgSrcinvDCT){
  vecImgSrcinvDCT.clear();
  for(int i = 0; i < 3; i++){
    Mat temp;
    dct(vecImgSrcDCT[i], temp, DCT_INVERSE);
    vecImgSrcinvDCT.push_back(temp);
  }
}

//=======================================================================================
// Display coef DCT
//=======================================================================================
void display_DCT(vector<Mat> &vecImgSrcDCT){
  int rows;
  int cols;

  vector<double*> maxValLoc;

  for(int k = 0; k < 3; k++){
    double maxVal;
    minMaxLoc(vecImgSrcDCT[k], NULL, &maxVal, NULL, NULL);

    rows = vecImgSrcDCT[k].rows;
    cols = vecImgSrcDCT[k].cols;

    for(int i = 0; i < rows; i++){
      for(int j = 0; j < cols; j++){
          vecImgSrcDCT[k].at<float>(i,j) = log(1+fabs(vecImgSrcDCT[k].at<float>(i,j)))/log(1+maxVal)*255;
      }
    }

    // Mat in(vecImgSrcDCT[k].size(), CV_8UC1);
    // vecImgSrcDCT[k].convertTo(in, CV_8UC1);
    // Mat out(vecImgSrcDCT[k].size(), CV_8UC1);
    // applyColorMap(in, out, COLORMAP_JET);
    // imshow("Coefficient DCT", norm_0_255(out));
    // waitKey();
  }
}

//=======================================================================================
// Appliquer mask sur DCT
//=======================================================================================
void applyMask(vector<Mat> & imgSrcDCT, int mask = 0)
{
  int x, y, width, height;

  int nbRows = imgSrcDCT[0].rows;
  int nbCols = imgSrcDCT[0].cols;

  Rect masque;

  if(mask == 0){
    x = (nbCols/2) + 10;
    y = (nbRows/2) + 10;
    width = nbCols - x;
    height = nbRows - y;

    std::cout << "X : " << x << " Y : " << y << " Width : " << width << " Height : " << height << std::endl;
    masque = Rect(x, y, width, height);
  }
  else if(mask == 1){
    x = (nbCols/2) + 10;
    y = 0;
    width = nbCols - x;
    height = nbRows - y;

    std::cout << "X : " << x << " Y : " << y << " Width : " << width << " Height : " << height << std::endl;
    masque = Rect(x, y, width, height);
  }
  else if(mask == 2){
    x = 0;
    y = (nbRows/2) + 10;
    width = nbCols - x;
    height = nbRows - y;

    std::cout << "X : " << x << " Y : " << y << " Width : " << width << " Height : " << height << std::endl;
    masque = Rect(x, y, width, height);
  }
  else if(mask == 3){
    x = 0;
    y = 0;
    width = nbCols - x;
    height = nbRows - y;

    std::cout << "X : " << x << " Y : " << y << " Width : " << width << " Height : " << height << std::endl;
    masque = Rect(x, y, width, height);
  }

  for(int k = 0; k < 3; k ++){
    //imgSrcDCT[k](masque).setTo(0);
  }
}

float MICD_1D(const Mat & I,int i, int j)
{
  if(j==0) return 128;
  else
    return I.at<float>(i,j-1);
}

float MICD_2D(const Mat & I,int i, int j)
{
  if(j==0 && i==0)  return 128;
  else if(j==0)     return (I.at<float>(i-1,j) + 128)/2;
  else if(i==0)     return (I.at<float>(i,j-1) + 128)/2;
  else              return (I.at<float>(i,j-1) + I.at<float>(i-1,j))/2;
}

float MICDA(const Mat & I,int i, int j)
{
  float a,b,c;
  if(j==0 && i==0)  return a=c=b=128;
  else if(j==0)     return a=b=128;
  else if(i==0)     return c=b=128;

  if(abs(c-b) < abs(a-b))
    return a;
  else
    return b;
}


void codage(const Mat & I,Mat & imgPrediction, int mode = 0, int q = 1)
{
  Mat imgDecodee(I.size(),CV_32FC1);

  float prediction;
  float dequantif;

  for (int i = 0; i < I.rows; ++i)
  {
    for (int j = 0; j < I.cols; ++j)
    {
      switch(mode){
        case 0: 
          prediction = MICD_1D(imgDecodee,i,j);
          break;
        
        case 1: 
          prediction = MICD_2D(imgDecodee,i,j);
          break;
        
        case 2: 
          prediction = MICDA(imgDecodee,i,j);
          break;
      }

      imgPrediction.at<float>(i,j) = floor( (I.at<float>(i,j) - prediction)/q + 0.5 );
      dequantif = imgPrediction.at<float>(i,j) * q;
      imgDecodee.at<float>(i,j) = dequantif + prediction;
    }
  }

  //return imgPrediction;
}


//=======================================================================================
//=======================================================================================
// MAIN
//=======================================================================================
//=======================================================================================
int main(int argc, char** argv){
  if (argc < 2){
    std::cout << "No image data or not enough ... At least one argument is required! \n";
    return -1;
  }

  Mat inputImageSrc;
  


  // Ouvrir l'image d'entr�e et v�rifier que l'ouverture du fichier se d�roule normalement
  inputImageSrc = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  if(!inputImageSrc.data ) { // Check for invalid input
        std::cout <<  "Could not open or find the image " << argv[1] << std::endl ;
		waitKey(0); // Wait for a keystroke in the window
        return -1;
  }

  vector<Mat> imgSrc;
  Mat img32FC(inputImageSrc.size(), CV_32FC3);

	//Conversion en YCrCb
  Mat inputImageSrcYCrCb;
	toYCrCb(inputImageSrc, inputImageSrcYCrCb);
  inputImageSrcYCrCb.convertTo(img32FC, CV_32FC3);

  split(img32FC, imgSrc);

    // Visualiser l'image
	imshow("inputImageSrcY", norm_0_255(imgSrc[0]));
	cvWaitKey();

  Mat predictionError(imgSrc[0].size(),CV_32FC1);
  codage(imgSrc[0], predictionError, 1);

  imshow("erreur de prediction", norm_0_255(predictionError));
  cvWaitKey();


  return 0;
}
