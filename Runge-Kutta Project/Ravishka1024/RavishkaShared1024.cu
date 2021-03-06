/*
Ravishka Rathnasuriya
GPU Project - 4th order Runge- Kutta algorithm
shared memory version for 1024 data sets using NVIDIA gtx device

Running the jobs
Copy the file RavishkaShared1024.c and shared1024 files
in your winscp folder.
using putty, go to the folder that contains above two files.
using:
"sbatch shared1024" command you can execute the program
and the .out file will be made in your folder with results. 
*/
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>

#define size 1024
#define threadsize 1024


//function prototypes
//for global memory and it will call the device diffOfy
__global__ void rk4thOrder(double *gx0,double *gx, double *gh, double *gy);
__device__ double diffOfy(double x, double y);

int main(){
    //data sets for x, x0, and h values
 double xsamples[8] = {0.3, 0.6,0.8,1.2, 1.5, 1.6,1.8,2.0};
 double x0samples[8] = {0.1, 0.2,0.4,0.3,0.5,0.4,0.6,0.5};
 double hsamples[8] = {0.1,0.2,0.4,0.3,0.3,0.4,0.3,0.5};
 

 //declaring memory for x0,x,h,y values dynamically

 double *x0 = (double *)malloc(size *sizeof(double));
 double *x = (double *)malloc(size *sizeof(double));
 double *h = (double *)malloc(size *sizeof(double));
 double *y = (double *)malloc(size *sizeof(double));

 //total bytes
 const int totalsize = size*sizeof(double);
 //declaring memory for global memory variables 
 double *Gx0; double *Gx; double *Gh;double *Gy; 
 cudaMalloc((void**)&Gx0,totalsize);
 cudaMalloc((void**)&Gx,totalsize);
 cudaMalloc((void**)&Gh,totalsize);
 cudaMalloc((void**)&Gy,totalsize);


   //for timing the intervals
   cudaEvent_t start, stop;
   //create two events start and stop
   cudaEventCreate(&start);
   cudaEventCreate(&stop);


//assigning values
//from 8th element onwards it will store same values
//for testing purposes.
 for(int i = 0; i< size; i++){
     if (i < 8){
         x[i] = xsamples[i];
         x0[i] = x0samples[i];
         h[i] = hsamples[i];
     }else{
     x[i]= 0.4;
     x0[i]= 0.0;
     h[i] = 0.1;
     }
 }

 //copying memory from host to device
//from x0,x, and h values to global memory
 cudaMemcpy(Gx0, x0, totalsize, cudaMemcpyHostToDevice);
 cudaMemcpy(Gx, x, totalsize, cudaMemcpyHostToDevice);
 cudaMemcpy(Gh, h, totalsize, cudaMemcpyHostToDevice);
 //cudaMemcpy(Gy, y, totalsize, cudaMemcpyHostToDevice);
 
//allocating number of blocks and threads in a block
 dim3 dimGrid(size/threadsize,1,1);
 dim3 dimBlock(threadsize,1,1);

  //start the time interval
cudaEventRecord(start);
rk4thOrder<<< dimGrid, dimBlock>>>(Gx0,Gx,Gh,Gy);
//end the timing
cudaEventRecord(stop);
//copy memory from device back to host 
cudaMemcpy(y, Gy, totalsize, cudaMemcpyDeviceToHost);

//wait until all device codes executes
cudaEventSynchronize(stop);
float milliseconds = 0.0;
//store the time difference in variable milliseconds
cudaEventElapsedTime(&milliseconds,start,stop);

//compute time in milli seconds,
printf("elapsed time is %lf milli secs \n",milliseconds);
//print the resulting values
for(int k = 0 ; k <size; k++){
printf("Answer at %dth position for dif eq is %.6lf \n",k, y[k]);
}
//freeing the memory 
cudaFree(Gx0);
cudaFree(Gx);
cudaFree(Gh);
cudaFree(Gh);


}


/*function rk4thoder - global function which is in the device
//parameters x0,y0,x,h,y 
return type double

the function will call diff eq function and compute the 4th order dif equation
iteration = (x -x0 /h)
k1, k2, k3, k4 will call diffofy function and iterate each of the time iteration results.*/

__global__
void rk4thOrder(double *gx0, double *gx, double *gh, double *gy){
   //invoking shared memory for xo,x,h,y variables. 
        double sx0[threadsize];
        double sx[threadsize];
        double sh[threadsize];
        double sy[threadsize];

    //copying data from global to shared memory using each iteration. 
        for(int m = 0; m < size/threadsize; m++){
         sx0[threadIdx.x] = gx0[m*threadsize + threadIdx.x];
         sx[threadIdx.x] = gx[m*threadsize + threadIdx.x];
        sh[threadIdx.x] = gh[m*threadsize + threadIdx.x];
         sy[threadIdx.x] = gy[m*threadsize + threadIdx.x];
         __syncthreads(); //wait till everything is done 

         //computing iteration numbers
        int iteration = ((sx[threadIdx.x]- sx0[threadIdx.x])/sh[threadIdx.x]);
       
        double k1, k2,k3,k4;
        sy[threadIdx.x] = 0.0;
        //for above iterations it will compute k1, k2,k3, k4 values and store in our new y value
        for(int j = 1; j <= iteration; j++){
            k1 = sh[threadIdx.x] * diffOfy(sx0[threadIdx.x],sy[threadIdx.x]);
            k2 = sh[threadIdx.x] * diffOfy((sx0[threadIdx.x]+ sh[threadIdx.x]/2), (sy[threadIdx.x] + k1/2));
            k3 = sh[threadIdx.x] * diffOfy((sx0[threadIdx.x] + sh[threadIdx.x]/2), (sy[threadIdx.x]+k2/2));
            k4 = sh[threadIdx.x]* diffOfy((sx0[threadIdx.x]+sh[threadIdx.x]), (sy[threadIdx.x]+k3));
            sy[threadIdx.x] += ((1.0/6.0)*(k1 +2*k2+2*k3+k4));
            sx0[threadIdx.x] += sh[threadIdx.x];
        }
        //copy the results from shared memory back to global memory
        gy[ m*threadsize + threadIdx.x] = sy[threadIdx.x];
        __syncthreads();
    }
}


//function name diffofy - device function
//parameters x, y
//return type double
//used a simple function that returns square of x and y summation
__device__
double diffOfy(double x, double y){
    return ((x*x)+ (y*y));
}