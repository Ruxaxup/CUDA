/*
 ============================================================================
 Name        : ProyectoCUDA.cu
 Author      : MTI
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals

 Cmd: nvcc `pkg-config --cflags opencv` ProyectoCUDA.cu `pkg-config --libs opencv` -o proyecto
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>


//Librerias de OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

#define IMG_WIDTH 256
#define IMG_HEIGHT 256
#define THREADS 100 //x 10
#define BLOCKS 66
#define IGUAL 1
#define DIFF 2
using namespace std;
using namespace cv;

bool isDiff(Mat A, Mat B);

/**
 * CUDA kernel that computes reciprocal values for a given vector
 */
__global__ void comparamela(unsigned char *d_MA,unsigned char *d_MB,unsigned char *d_MC) {
	int id = blockIdx.x * blockDim.x * blockDim.y
				+ threadIdx.y * blockDim.x + threadIdx.x;

	if(id < IMG_WIDTH*IMG_WIDTH)
		d_MC[id] = d_MA[id] - d_MB[id];
}

int main(int argc, char *argv[])
{
	unsigned char matrizA[IMG_WIDTH * IMG_HEIGHT];
	unsigned char matrizB[IMG_WIDTH * IMG_HEIGHT];
	unsigned char matrizC[IMG_WIDTH * IMG_HEIGHT];

	unsigned char *d_MA, *d_MB, *d_MC;

	//Minimo y maximo dos argumentos
	if(argc > 3){
		printf("El numero de argumentos debe ser igual a 3\n.");
		return 3;
	}

	//Cargar imagenes
	char *name1 = argv[1];
	char *name2 = argv[2];

	Mat imageA;
	Mat imageB;
	
	imageA = imread(name1,1);
	imageB = imread(name2,1);


	namedWindow("Display Image", WINDOW_AUTOSIZE );
	imshow("Display Image", imageA);
	namedWindow("Display Image2", WINDOW_AUTOSIZE );
	imshow("Display Image2", imageB);

	/*
		VErificar dimensiones de las imagenes
	*/
	if(isDiff(imageA,imageB))
	{
		printf("**Las dimensiones no coinciden.\n");
		return DIFF;
	}

	//////////////////////
	//Copiar imagenes a arreglos
	Vec3b intensityA,intensityB;
	for(int i=0; i<IMG_WIDTH; i++){
		for(int j=0; j<IMG_WIDTH; j++){
		    intensityA = imageA.at<Vec3b>(i, j);
		    intensityB = imageB.at<Vec3b>(i, j);
		    matrizA[i*IMG_WIDTH+j]=(unsigned char)intensityA.val[2];
		    matrizB[i*IMG_WIDTH+j]=(unsigned char)intensityB.val[2];
		    matrizC[i*IMG_WIDTH+j] = 0;
		}
	}
	
	cudaMalloc((void**)&d_MA,sizeof(char)*IMG_HEIGHT*IMG_WIDTH);
	cudaMalloc((void**)&d_MB,sizeof(char)*IMG_HEIGHT*IMG_WIDTH);
	cudaMalloc((void**)&d_MC,sizeof(char)*IMG_HEIGHT*IMG_WIDTH);

	cudaMemcpy(d_MA,matrizA,sizeof(char)*IMG_WIDTH*IMG_HEIGHT,cudaMemcpyHostToDevice);
	cudaMemcpy(d_MB,matrizB,sizeof(char)*IMG_WIDTH*IMG_HEIGHT,cudaMemcpyHostToDevice);

	dim3 bloque(BLOCKS);
	dim3 hilos(10,THREADS);

	comparamela<<<bloque,hilos>>>(d_MA,d_MB,d_MC);



	cudaMemcpy(matrizC,d_MC,sizeof(char)*IMG_HEIGHT*IMG_WIDTH,cudaMemcpyDeviceToHost);

	//Verificar si son iguales
	int diferente = IGUAL;
	for(int i=0; i<IMG_WIDTH; i++){
		for(int j=0; j<IMG_WIDTH; j++){
			if(matrizC[i*IMG_WIDTH+j] != 0){
				diferente = DIFF;
				break;
			}					    
		}
		if(diferente) break;
	}

	Mat imagedif;
	imagedif = imageB;
	Vec3b intensityC;
	//Se recrea la imagen a partir del arreglo c
    for(int i=0; i<IMG_WIDTH; i++){
		for(int j=0; j<IMG_WIDTH; j++){
		    intensityC.val[0] = matrizC[i*IMG_WIDTH+j];
		    intensityC.val[1] = matrizC[i*IMG_WIDTH+j];
		    intensityC.val[2] = matrizC[i*IMG_WIDTH+j];
		    imagedif.at<Vec3b>(i, j)=intensityC;
		}
    }
    namedWindow( "diferencia de im", CV_WINDOW_NORMAL );
    imshow( "diferencia de im", imagedif);
	waitKey(0);

	/* Free memory */
	cudaFree(d_MA);
	cudaFree(d_MB);
	cudaFree(d_MC);
	return diferente;
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

bool isDiff(Mat A, Mat B)
{
	return (A.rows != B.rows) && (A.cols != B.cols);
}