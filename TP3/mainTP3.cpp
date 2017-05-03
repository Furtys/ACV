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
float eqm(const Mat & img1, const Mat & img2)
{
	int rows = img1.rows;
	int cols = img1.cols;

	if(rows != img2.rows || cols != img2.cols){
		cout << "Error, images need to have the same size. \n";
	}

	float EQM = 0;

	for(int i = 0; i < rows; i++){
		for(int j = 0; j < cols; j++){
	      float pixel1 = img1.at<float>(i, j);
	      float pixel2 = img2.at<float>(i, j);
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

	float EQM = eqm(imgSrc, imgDCT);

	if(EQM == 0){
		cout << "Error, PSNR can't compare two identic images \n";
		return -1;
	}
	float psnr = 10 * log10(65025/EQM);
	cout << "PSNR : " << psnr << "\n";
 	return psnr;
}

//=======================================================================================
// Predicteur
//=======================================================================================

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
	float a, b, c;

	if(j==0 && i==0)
	{
		a=b=c=128;
	}  
  	else if(j==0) 
  	{
  		b=c=128;
  		a=I.at<float>(i-1,j);
  	}    
  	else if(i==0)
  	{
  		a=b=128;
  		c=I.at<float>(i,j-1);
  	}     
  	else  
  	{
  		a=I.at<float>(i-1,j);
  		b=I.at<float>(i-1,j-1);
  		c=I.at<float>(i,j-1);
  	}    

  	if(abs(c-b) < abs(a-b))
		return a;
	else
	    return b;        
}

//=======================================================================================
// Codage
//=======================================================================================

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

  	for (int i = 0; i < I.rows; ++i)
   	{
   		for (int j = 0; j < I.cols; ++j)
    	{
    		imgPrediction.at<float>(i,j) += 128;
    	}
	}
}

void codage_competitif(const Mat & I,Mat & imgPrediction, Mat & imgPredicteur, int q = 1)
{
	Mat imgDecodee(I.size(),CV_32FC1);

	float prediction;
	float dequantif;
	float a, b, c, val,aa ,bb, cc;

	for (int i = 0; i < I.rows; ++i)
  	{
  		for (int j = 0; j < I.cols; ++j)
    	{
	    	aa = MICD_1D(imgDecodee,i,j);
	    	bb = MICD_2D(imgDecodee,i,j);
	    	cc = MICDA(imgDecodee,i,j);
	    	val = I.at<float>(i,j);
	    	a = abs(val - aa);
	    	b = abs(val - bb);
	    	c = abs(val - cc);
	    	
	    	if(a < b && a < c) 
	    	{
	    		imgPredicteur.at<float>(i,j) = 0;
	    		prediction = aa;
	    	}
	    	else if(b < c)
	    	{
	    		imgPredicteur.at<float>(i,j) = 1;
	    		prediction = bb;
	    	}
	    	else
	    	{
	    		imgPredicteur.at<float>(i,j) = 2;
	    		prediction = cc;
	    	}
    
		    imgPrediction.at<float>(i,j) = floor( (I.at<float>(i,j) - prediction)/q + 0.5 );
		    dequantif = imgPrediction.at<float>(i,j) * q;
		    imgDecodee.at<float>(i,j) = dequantif + prediction;
		}
   	}

   	for (int i = 0; i < I.rows; ++i)
   	{
   		for (int j = 0; j < I.cols; ++j)
    	{
    		imgPrediction.at<float>(i,j) += 128;
    	}
	}
}
//=======================================================================================
// Décodage
//=======================================================================================

void decodage(const Mat & imgPrediction,Mat & I, int mode = 0, int q = 1)
{
  	float prediction;
  	float dequantif;

  	for (int i = 0; i < I.rows; ++i)
  	{
    	for (int j = 0; j < I.cols; ++j)
    	{
      		switch(mode){
		        case 0: 
		          prediction = MICD_1D(I,i,j);
		          break;
		        
		        case 1: 
		          prediction = MICD_2D(I,i,j);
		          break;
		        
		        case 2: 
		          prediction = MICDA(I,i,j);
		          break;
      		}

	    	dequantif = (imgPrediction.at<float>(i,j) - 128) * q;
	    	I.at<float>(i,j) = dequantif + prediction;
    	}
  	}
}

void decodage_competitif(const Mat & imgPrediction, const Mat & imgPredicteur, Mat & I, int q = 1)
{
	float prediction;
  	float dequantif;

  	for (int i = 0; i < I.rows; ++i)
  	{
    	for (int j = 0; j < I.cols; ++j)
    	{
			if(imgPredicteur.at<float>(i,j) == 0) 
			{
				prediction = MICD_1D(I,i,j);
			}
			else if(imgPredicteur.at<float>(i,j) == 1)
			{
				prediction = MICD_2D(I,i,j);
			}
			else
			{
				prediction = MICDA(I,i,j);
			}
		    
		    dequantif = (imgPrediction.at<float>(i,j) - 128) * q;
			I.at<float>(i,j) = dequantif + prediction;
		}
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
  Mat predictor(imgSrc[0].size(),CV_32FC1);
  Mat decodee(imgSrc[0].size(),CV_32FC1);
  Mat histo;
  
//=======================================================================================
// MICD_1D
//=======================================================================================
std::cout << "\n--------------- MICD_1D ---------------" << std::endl;
  codage(imgSrc[0], predictionError, 0, 8);
  computeHistogram(predictionError, histo);
  entropyCalculus(predictionError, histo);
  decodage(predictionError, decodee, 0, 8);
  psnr(imgSrc[0],decodee);

  imshow("erreur de prediction", norm_0_255(predictionError));
  imshow("image decodee", norm_0_255(decodee));
  displayHistogram(histo);
  imwrite((string)argv[1] + "_errorMap_MICD_1D", norm_0_255(predictionError));
  cvWaitKey();

//=======================================================================================
// MICD_2D
//=======================================================================================
std::cout << "\n--------------- MICD_2D ---------------" << std::endl;
  codage(imgSrc[0], predictionError, 1, 8);
  computeHistogram(predictionError, histo);
  entropyCalculus(predictionError, histo);
  decodage(predictionError, decodee, 1, 8);
  psnr(imgSrc[0],decodee);

  imshow("erreur de prediction", norm_0_255(predictionError));
  imshow("image decodee", norm_0_255(decodee));
  displayHistogram(histo);
  imwrite((string)argv[1] + "_errorMap_MICD_2D", norm_0_255(predictionError));
  cvWaitKey();

//=======================================================================================
// MICDA
//=======================================================================================
std::cout << "\n--------------- MICDA ---------------" << std::endl;
  codage(imgSrc[0], predictionError, 2, 8);
  computeHistogram(predictionError, histo);
  entropyCalculus(predictionError, histo);
  decodage(predictionError, decodee, 2, 8);
  psnr(imgSrc[0],decodee);

  imshow("erreur de prediction", norm_0_255(predictionError));
  imshow("image decodee", norm_0_255(decodee));
  displayHistogram(histo);
  imwrite((string)argv[1] + "_errorMap_MICDA", norm_0_255(predictionError));
  cvWaitKey();

//=======================================================================================
// Competitif
//=======================================================================================
std::cout << "\n--------------- Compet ---------------" << std::endl;
  codage_competitif(imgSrc[0], predictionError, predictor, 8);
  computeHistogram(predictionError, histo);
  entropyCalculus(predictionError, histo);
  decodage_competitif(predictionError, predictor, decodee, 8);
  psnr(imgSrc[0],decodee);

  imshow("erreur de prediction", norm_0_255(predictionError));
  imshow("Predicteur choisi", norm_0_255(predictor));
  imshow("image decodee", norm_0_255(decodee));
  imwrite((string)argv[1] + "_errorMap_compet", norm_0_255(predictionError));
  imwrite((string)argv[1] + "_predictor_choice", norm_0_255(predictor));
  cvWaitKey();


  return 0;
}
