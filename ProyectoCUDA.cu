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
#include <fstream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


//Librerias de OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

#define TIME_FILE "elapsedTime.txt"
#define THREADS 32 //x 32
#define BLOCKS 66
#define IGUAL 0
#define DIFF 1
using namespace std;
using namespace cv;

bool isEq(Mat A, Mat B);
void writeTime(float elapsedTime);

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
	unsigned char *matrizA = NULL;
	unsigned char *matrizB = NULL;
	unsigned char *matrizC = NULL;
	unsigned char *d_MA, *d_MB, *d_MC;

	int resolution;
	unsigned int threads_x, threads_y;
	unsigned int blocks_x, blocks_y;
	int width, height;

	width = height = 1;
	threads_x = threads_y 	= 1;
	blocks_x = blocks_y = 1;

	//Variables control de tiempo
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

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
	//cout << "[" << name1 << "] -> [" << name2 << "]" << endl;
	/*
		Verificar dimensiones de las imagenes
	*/
	if(!isEq(imageA,imageB))
	{
		//printf("**Las dimensiones no coinciden.\n");
		return DIFF;
	}
	width = imageA.cols;
	height = imageA.rows;
	resolution = width * height;

	//Reserva de memoria en el host
	//cout << "***RESERVA DE MEMORIA: "<<resolution<< endl;
	matrizA = (unsigned char*)malloc(sizeof(unsigned char) * resolution);
	matrizB = (unsigned char*)malloc(sizeof(unsigned char) * resolution);
	matrizC = (unsigned char*)malloc(sizeof(unsigned char) * resolution);

	/*matrizA = new unsigned char[resolution];
	matrizB = new unsigned char[resolution];
	matrizC = new unsigned char[resolution];*/

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
	cudaMalloc((void**)&d_MA,sizeof(unsigned char)*resolution);
	cudaMalloc((void**)&d_MB,sizeof(unsigned char)*resolution);
	cudaMalloc((void**)&d_MC,sizeof(unsigned char)*resolution);

	//Copia imagenes a device
	cudaMemcpy(d_MA,matrizA,sizeof(unsigned char)*width*height,cudaMemcpyHostToDevice);
	cudaMemcpy(d_MB,matrizB,sizeof(unsigned char)*width*height,cudaMemcpyHostToDevice);


	//Asignacion de bloques
	if(imageA.rows < 32 && imageA.cols < 32)
	{
		threads_x = imageA.rows;
		threads_y = imageA.cols;
	}
	else
	{
		threads_x = threads_y = 32;
		blocks_x = ceil(imageA.cols / threads_x);
		blocks_y = ceil(imageA.rows / threads_y);
	}

	dim3 bloque(blocks_x, blocks_y);
	dim3 hilos(threads_x, threads_y);

	//Tomamos tiempo inicial
	cudaEventRecord(start, 0);

	//Lanzamiento del kernel
	comparamela<<<bloque,hilos>>>(d_MA,d_MB,d_MC, resolution);
	//Copia resultado al host
	cudaMemcpy(matrizC,d_MC,sizeof(unsigned char)*width*height,cudaMemcpyDeviceToHost);

	//cout << "***COMPARACION DE MEMORIA"<< endl;
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
	//cout << "***LIBERACION DE MEMORIA"<< endl;

	//Detenemos el tiempo
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	//Calculamos tiempo
	cudaEventElapsedTime(&elapsedTime,start,stop);
	writeTime(elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	/* Free memory */
	cudaFree(d_MA);
	cudaFree(d_MB);
	cudaFree(d_MC);

	free(matrizA);
	free(matrizB);
	free(matrizC);
	//cout << "***LIBERACION DE MEMORIA COMPLETADA"<< endl;
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

bool isEq(Mat A, Mat B)
{
	//cout << "A: " <<A.cols<<","<<A.rows<<endl;
	//cout << "B: " <<B.cols<<","<<B.rows<<endl;
	return (A.rows == B.rows) && (A.cols == B.cols);
}

void writeTime(float elapsedTime)
{
	ofstream fs(TIME_FILE, ofstream::out | ofstream::app);
	fs << elapsedTime << endl;
	fs.close();
}