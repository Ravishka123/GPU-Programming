/*
Ravishka Rathnasuriya
GPU Project - 4th order Runge- Kutta algorithm
Serial version for 1024 data sets using NVIDIA gtx device

Running the jobs
Copy the file RavishkaSerial1024.c and serial1024 files
in your winscp folder.
using putty, go to the folder that contains above two files.
using:
"sbatch serial1024" command you can execute the program
and the .out file will be made in your folder with results. 
*/


#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>

#define size 1024
#define billion 1000000000L

//function prototypes
double rk4thOrder(double *x0, double y0,double *x, double *h, double *y);
double diffOfy(double x, double y);

int main(){
    //data sets for x, x0, and h values
 double xsamples[8] = {0.3, 0.6,0.8,1.2, 1.5, 1.6,1.8,2.0};
 double x0samples[8] = {0.1, 0.2,0.4,0.3,0.5,0.4,0.6,0.5};
 double hsamples[8] = {0.1,0.2,0.4,0.3,0.3,0.4,0.3,0.5};
 
 //compute time, initialization
 uint64_t diff;
 struct timespec start, end;

//used value 0 for y0
 double y0= 0.0;

//declaring memory for x0,x,h,y values dynamically
 double *x0 = (double *)malloc(size *sizeof(double));
 double *x = (double *)malloc(size *sizeof(double));
 double *h = (double *)malloc(size *sizeof(double));
 double *y = (double *)malloc(size *sizeof(double));

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

//start the clock and call the necessary function
clock_gettime(CLOCK_MONOTONIC, &start);
//the function will compute the runge-kutta 4th order dif equation
rk4thOrder(x0,y0,x,h,y);
//end the clock
clock_gettime(CLOCK_MONOTONIC, &end);
//compute time in milli seconds, nano seconds / 1000000
diff = billion *(end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
printf(" elapsed time = %lf milli seconds\n", diff/1000000.0);

//print the resulting values 
for(int k = 0 ; k <size; k++){
printf("Answer at %dth position for dif eq is %.6lf \n",k, y[k]);
}

}

/*function rk4thoder
//parameters x0,y0,x,h,y 
return type double

the function will call diff eq function and compute the 4th order dif equation
iteration = (x -x0 /h)
k1, k2, k3, k4 will call diffofy function and iterate each of the time iteration results.*/


double rk4thOrder(double *x0, double y0,double *x, double *h, double *y){
    for(int i  =0; i< size; i++){
        int iteration = ((x[i]- x0[i])/h[i]); //computing iteration numbers
       
        double k1, k2,k3,k4;
        y[i] = y0;
       
       //for above iterations it will compute k1, k2,k3, k4 values and store in our new y value
        for(int j = 1; j <= iteration; j++){
            k1 = h[i] * diffOfy(x0[i],y[i]);
            k2 = h[i] * diffOfy((x0[i]+ h[i]/2), (y[i] + k1/2));
            k3 = h[i] * diffOfy((x0[i] + h[i]/2), (y[i]+k2/2));
            k4 = h[i]* diffOfy((x0[i]+h[i]), (y[i]+k3));
            y[i] += ((1.0/6.0)*(k1 +2*k2+2*k3+k4));
            x0[i] += h[i];
        }
    }

}


//function name diffofy
//parameters x, y
//return type double
//used a simple function that returns square of x and y summation
double diffOfy(double x, double y){
    return ((x*x)+ (y*y));
}