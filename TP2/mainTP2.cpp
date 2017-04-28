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
// to BGR
//=======================================================================================
void toBGR(const Mat & imgSrcYCrCb, Mat &ImgSrc){
	cvtColor(imgSrcYCrCb, ImgSrc, CV_YCrCb2BGR);
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
void DCT(const vector<Mat> &vecImgSrc, vector<Mat> &vecImgSrcDCT, bool inverse = false){
  vecImgSrcDCT.clear();
  for(int k = 0; k < 3; k++){
    Mat temp(vecImgSrc[k].size(), CV_32FC1);
    if(!inverse){
      dct(vecImgSrc[k], temp);
    }
    if(inverse){
      dct(vecImgSrc[k], temp, DCT_INVERSE);
    }
    vecImgSrcDCT.push_back(temp);
  }
}

//=======================================================================================
// DCT Inverse
//=======================================================================================
void inverseDCT(const vector<Mat> &vecImgSrcDCT, vector<Mat> &vecImgSrcinvDCT){
  DCT(vecImgSrcDCT, vecImgSrcinvDCT, true);
}

//=======================================================================================
// blockDCT
//=======================================================================================
void blockDCT(const vector<Mat> &vecImgSrc, vector<Mat> &vecImgSrcDCT, bool inverse = false){
  for(int k = 0; k < 3; k++){
    Mat temp(vecImgSrc[k].size(), CV_32FC1);

    for(int i = 0; i < vecImgSrc[k].rows; i+=8){
      for(int j = 0; j < vecImgSrc[k].cols; j+=8){
        Rect window(i, j, 8, 8);
        Mat block = vecImgSrc[k](window);

        if(!inverse){
          dct(block, temp(window));
        }
        if(inverse){
          dct(block, temp(window), DCT_INVERSE);
        }

      }
    }
    vecImgSrcDCT.push_back(temp);
  }
}

//=======================================================================================
// blockDCT
//=======================================================================================
void inverseblockDCT(const vector<Mat> &vecImgSrc, vector<Mat> &vecImgSrcDCT){
  blockDCT(vecImgSrc, vecImgSrcDCT, true);
}
//=======================================================================================
// Display coef DCT
//=======================================================================================
void display_DCT(const vector<Mat> &vecImgSrcDCT){
  int rows;
  int cols;

  vector<double*> maxValLoc;

  for(int k = 0; k < 3; k++){
    double maxVal;
    minMaxLoc(vecImgSrcDCT[k], NULL, &maxVal, NULL, NULL);

    Mat res(vecImgSrcDCT[k].size(), CV_32FC1);

    rows = vecImgSrcDCT[k].rows;
    cols = vecImgSrcDCT[k].cols;

    for(int i = 0; i < rows; i++){
      for(int j = 0; j < cols; j++){
          res.at<float>(i,j) = log(1+fabs(vecImgSrcDCT[k].at<float>(i,j)))/log(1+maxVal)*255;
      }
    }

    // Mat in(vecImgSrcDCT[k].size(), CV_8UC1);
    // vecImgSrcDCT[k].convertTo(in, CV_8UC1);
    // Mat out(vecImgSrcDCT[k].size(), CV_8UC1);
    // applyColorMap(in, out, COLORMAP_JET);
    // imshow("Coefficient DCT", norm_0_255(out));
    // waitKey();

    imshow("Coefficients DCT", norm_0_255(res));
    waitKey();
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
    x = (nbCols/2);
    y = (nbRows/2);
    width = nbCols - x;
    height = nbRows - y;

    std::cout << "X : " << x << " Y : " << y << " Width : " << width << " Height : " << height << std::endl;
    masque = Rect(x, y, width, height);
  }
  else if(mask == 1){
    x = (nbCols/2);
    y = 0;
    width = nbCols - x;
    height = nbRows - y;

    std::cout << "X : " << x << " Y : " << y << " Width : " << width << " Height : " << height << std::endl;
    masque = Rect(x, y, width, height);
  }
  else if(mask == 2){
    x = 0;
    y = (nbRows/2);
    width = nbCols - x;
    height = nbRows - y;

    std::cout << "X : " << x << " Y : " << y << " Width : " << width << " Height : " << height << std::endl;
    masque = Rect(x, y, width, height);
  }
  else if(mask == 3){
    x = 0;
    y = (nbRows/2);
    width = (nbCols/2);
    height = nbRows - y;

    std::cout << "X : " << x << " Y : " << y << " Width : " << width << " Height : " << height << std::endl;
    masque = Rect(x, y, width, height);
  }

  for(int k = 0; k < 3; k ++){
    imgSrcDCT[k](masque).setTo(0);
  }
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
  vector<Mat> imgSrcDCT;
  vector<Mat> imgSrcInverseDCT;
  Mat img32FC(inputImageSrc.size(), CV_32FC3);

	//Conversion en YCrCb
  Mat inputImageSrcYCrCb;
	toYCrCb(inputImageSrc, inputImageSrcYCrCb);
  inputImageSrcYCrCb.convertTo(img32FC, CV_32FC3);

  split(img32FC, imgSrc);

    // Visualiser l'image
	imshow("inputImageSrcY", norm_0_255(imgSrc[0]));
	cvWaitKey();

  //calcul de la DCT de imgSrc
  cout << "----------- COMPUTE DCT --------------" << endl;
  DCT(imgSrc, imgSrcDCT);
  //Calcul de la DCT inverse de imgSrcDCT
  cout << "----------- COMPUTE inverseDCT --------------" << endl;
  inverseDCT(imgSrcDCT, imgSrcInverseDCT);
  imshow("Inverse DCT Y", norm_0_255(imgSrcInverseDCT[0]));
  imshow("Inverse DCT Cr", norm_0_255(imgSrcInverseDCT[1]));
  imshow("Inverse DCT Cb", norm_0_255(imgSrcInverseDCT[2]));
  waitKey();

	cout << "Calculs between " << argv[1] << "_Y and inverseDCT_Y : "<< "\n";
	psnr(imgSrc[0], imgSrcInverseDCT[0]);
	cout << "Calculs between " << argv[1] << "_Cr and inverseDCT_Cr : "<< "\n";
	psnr(imgSrc[1], imgSrcInverseDCT[1]);
	cout << "Calculs between " << argv[1] << "_Cb and inverseDCT_Cb : "<< "\n";
	psnr(imgSrc[2], imgSrcInverseDCT[2]);


  //Affichage des coefficients DCT et de l'entropie des coefficients
  //vector<Mat> imgSrcDCTDisplayed = imgSrcDCT;
  display_DCT(imgSrcDCT);
  vector<Mat> histosCoef;
  /*std::cout << "Entropie des coefficients" << std::endl;
  for(int i = 0; i < 3; i ++){
    imshow("Coefficient DCT", norm_0_255(imgSrcDCTDisplayed[i]));
    waitKey();
    Mat tempHist;
    computeHistogram(imgSrcDCT[i], tempHist);
    histosCoef.push_back(displayHistogram(tempHist));
    entropyCalculus(imgSrcDCT[i], tempHist);
  }*/

  //Calcul entropie image source
/*  vector<Mat> histosSrc;
  std::cout << "Entropie de l'image source" << std::endl;
  for(int i = 0; i < 3; i ++){
    imshow("Image originale", norm_0_255(imgSrc[i]));
    waitKey();
    Mat tempHist;
    computeHistogram(imgSrc[i], tempHist);
    histosSrc.push_back(displayHistogram(tempHist));
    entropyCalculus(imgSrc[i], tempHist);
  }*/

  //2.3 - Annulation des coefficients DCT
  std::cout << "Annulation des coefficients" << std::endl;
  //
  applyMask(imgSrcDCT, 0);
  display_DCT(imgSrcDCT);
  inverseDCT(imgSrcDCT, imgSrcInverseDCT);
  std::cout << "PSNR image source et DCT erased" << std::endl;
  psnr(imgSrc[0], imgSrcInverseDCT[0]);
  imshow("Inverse DCT Y_erased", norm_0_255(imgSrcInverseDCT[0]));
  waitKey();
  imshow("Inverse DCT Cr_erased", norm_0_255(imgSrcInverseDCT[1]));
  waitKey();
  imshow("Inverse DCT Cb_erased", norm_0_255(imgSrcInverseDCT[2]));
  waitKey();

  //Fusions des canaux
  Mat finalImg32F;
  merge(imgSrcInverseDCT, finalImg32F);
  //Conversion en uchar pour affichage
  Mat finalImg;
  finalImg32F.convertTo(finalImg, CV_8UC3);
  //Conversion en BGR
  toBGR(finalImg, finalImg);
  imshow("Final Image", finalImg);
  waitKey();

  //3 - DCT Block 8x8
  imgSrcDCT.clear();
  blockDCT(imgSrc, imgSrcDCT);
  display_DCT(imgSrcDCT);
  inverseblockDCT(imgSrcDCT, imgSrcInverseDCT);
  imshow("Inverse DCT Y block", norm_0_255(imgSrcInverseDCT[0]));
  waitKey();
  imshow("Inverse DCT Cr block", norm_0_255(imgSrcInverseDCT[1]));
  waitKey();
  imshow("Inverse DCT Cb block", norm_0_255(imgSrcInverseDCT[2]));
  waitKey();

  Mat_<float> m1(8, 8);
  m1 << 16, 11, 10, 16, 24, 40, 51, 61,
      12, 12, 14, 19, 26, 58, 60, 55,
      14, 13, 16, 24, 40, 57, 69, 56,
      14, 17, 22, 29, 51, 87, 80, 62,
      18, 22, 37, 56, 68, 109, 103, 77,
      24, 35, 55, 64, 81, 104, 113, 92,
      49, 64, 78, 87, 103, 121, 120, 101,
      72, 92, 95, 98, 112, 100, 103, 99;
  Mat coefJPEG = m1;

  return 0;
}
