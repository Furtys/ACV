#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

using namespace cv;
using namespace std;


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
// to YCrCb
//=======================================================================================
void toYCrCb(const Mat & imgSrc, Mat &ImgSrcYCrCb){
	cvtColor(imgSrc, ImgSrcYCrCb, CV_BGR2YCrCb);
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
  for(int i = 0; i < 3; i++){
    Mat temp;
    dct(vecImgSrcDCT[i], temp, DCT_INVERSE);
    vecImgSrcinvDCT.push_back(temp);
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
  waitKey();

	cout << "Calculs between " << argv[1] << "_Y and inverseDCT_Y : "<< "\n";
	psnr(imgSrc[0], imgSrcInverseDCT[0]);
	cout << "Calculs between " << argv[1] << "_Cr and inverseDCT_Cr : "<< "\n";
	psnr(imgSrc[1], imgSrcInverseDCT[1]);
	cout << "Calculs between " << argv[1] << "_Cb and inverseDCT_Cb : "<< "\n";
	psnr(imgSrc[2], imgSrcInverseDCT[2]);

  return 0;
}
