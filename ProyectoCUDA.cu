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

#define THREADS 32 //x 32
#define BLOCKS 66
#define IGUAL 0
#define DIFF 1
using namespace std;
using namespace cv;

bool isDiff(Mat A, Mat B);


__global__ void comparamela(unsigned char *d_MA,unsigned char *d_MB,unsigned char *d_MC, unsigned int resolution) {
	/*int id = blockIdx.x * blockDim.x * blockDim.y
				+ threadIdx.y * blockDim.x + threadIdx.x;*/
	int blockId = blockIdx.x
		+ blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y)
		+ (threadIdx.y * blockDim.x)
		+ threadIdx.x;

	if(threadId < resolution)
		d_MC[threadId] = d_MA[threadId] - d_MB[threadId];
}

int main(int argc, char *argv[])
{
	unsigned char *matrizA;
	unsigned char *matrizB;
	unsigned char *matrizC;
	unsigned char *d_MA, *d_MB, *d_MC;

	unsigned int resolution;
	unsigned int threads_x, threads_y;
	unsigned int blocks_x, blocks_y;
	unsigned int width, height;

	threads_x = threads_y 	= 1;
	blocks_x = blocks_y = 1;

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

	/*
		Verificar dimensiones de las imagenes
	*/
	if(isDiff(imageA,imageB))
	{
		printf("**Las dimensiones no coinciden.\n");
		return DIFF;
	}
	width = imageA.cols;
	height = imageA.rows;
	resolution = width * height;

	//Reserva de memoria en el host
	matrizA = (unsigned char*)malloc(sizeof(unsigned char) * width * height);
	matrizB = (unsigned char*)malloc(sizeof(unsigned char) * width * height);
	matrizC = (unsigned char*)malloc(sizeof(unsigned char) * width * height);

	//Copiar imagenes a arreglos
	Vec3b intensityA,intensityB;
	for(int i=0; i<width; i++){
		for(int j=0; j<height; j++){
		    intensityA = imageA.at<Vec3b>(i, j);
		    intensityB = imageB.at<Vec3b>(i, j);
		    matrizA[i*width+j]=(unsigned char)intensityA.val[2];
		    matrizB[i*width+j]=(unsigned char)intensityB.val[2];
		    matrizC[i*width+j] = 0;
		}
	}
	
	//Reserva de memoria en el device
	cudaMalloc((void**)&d_MA,sizeof(char)*resolution);
	cudaMalloc((void**)&d_MB,sizeof(char)*resolution);
	cudaMalloc((void**)&d_MC,sizeof(char)*resolution);

	//Copia imagenes a device
	cudaMemcpy(d_MA,matrizA,sizeof(char)*width*height,cudaMemcpyHostToDevice);
	cudaMemcpy(d_MB,matrizB,sizeof(char)*width*height,cudaMemcpyHostToDevice);


	//Asignacion de bloques
	if(imageA.rows < 32 && imageA.cols < 32)
	{
		threads_x = imageA.rows;
		threads_y = imageA.cols;
	}

	dim3 bloque(blocks_x, blocks_y);
	dim3 hilos(threads_x, threads_y);

	//Lanzamiento del kernel
	comparamela<<<bloque,hilos>>>(d_MA,d_MB,d_MC, resolution);
	//Copia resultado al host
	cudaMemcpy(matrizC,d_MC,sizeof(char)*width*height,cudaMemcpyDeviceToHost);

	//Verificar si son iguales
	int diferente = IGUAL;
	for(int i=0; i<width; i++){
		for(int j=0; j<height; j++){
			if(matrizC[i*width+j] != 0){
				diferente = DIFF;
				break;
			}					    
		}
		if(diferente) break;
	}

	/* Free memory */
	cudaFree(d_MA);
	cudaFree(d_MB);
	cudaFree(d_MC);

	free(matrizA);
	free(matrizB);
	free(matrizC);
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
