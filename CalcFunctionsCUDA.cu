//Copyright (C) 2015, NRC "Kurchatov institute", http://www.nrcki.ru/e/engl.html, Moscow, Russia
//Author: Vladislav Neverov, vs-never@hotmail.com, neverov_vs@nrcki.ru
//
//This file is part of XaNSoNS.
//
//XaNSoNS is free software: you can redistribute it and / or modify
//it under the terms of the GNU General Public License as published by
//the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.
//
//XaNSoNS is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//GNU General Public License for more details.
//
//You should have received a copy of the GNU General Public License
//along with this program. If not, see <http://www.gnu.org/licenses/>.

#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "typedefs.h"
#ifdef UseCUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
void calcRotMatrix(vect3d <double> *cf0, vect3d <double> *cf1, vect3d <double> *cf2, vect3d <double> euler, unsigned int convention);
//some float4 and float 3 functions (float4 used as float3)
inline __device__ __host__ float dot(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline __device__ __host__ float dot(float3 a, float4 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline __device__ __host__ float dot(float4 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline __host__ __device__ float3 operator+(float3 a, float3 b){ return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline __host__ __device__ float3 operator-(float3 a, float3 b){ return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline __host__ __device__ float3 operator*(float3 a, float b){ return make_float3(a.x * b, a.y * b, a.z * b); }
inline __device__ float length(float3 v){ return sqrtf(dot(v, v)); }
unsigned int GetGFLOPS(cudaDeviceProp deviceProp, bool show = false){
	//estimating the theoretical GFLOPS
	unsigned int cc = deviceProp.major * 10 + deviceProp.minor; //compute capability
	unsigned int MP = deviceProp.multiProcessorCount; //number of multiprocessors
	unsigned int clockRate = deviceProp.clockRate / 1000; //GPU clockrate
	unsigned int GFLOPS = MP * 128 * 2 * clockRate / 1000; 
	switch (cc){
	case 10:
	case 11:
	case 12:
	case 13:
		GFLOPS = MP * 8 * 2 * clockRate / 1000;
		break;
	case 20:
		GFLOPS = MP * 32 * 2 * clockRate / 1000;
		break;
	case 21:
		GFLOPS = MP * 48 * 2 * clockRate / 1000;
		break;
	case 30:
	case 35:
	case 37:
		GFLOPS = MP * 192 * 2 * clockRate / 1000;
		break;
	case 50:
	case 52:
	case 61:
		GFLOPS = MP * 128 * 2 * clockRate / 1000;
		break;
	case 60:
		GFLOPS = MP * 64 * 2 * clockRate / 1000;
		break;	
	}
	if (show) {
		cout << "GPU name: " << deviceProp.name << "\n";
		cout << "CUDA compute capability: " << deviceProp.major << "." << deviceProp.minor << "\n";
		cout << "Number of multiprocessors: " << MP << "\n";
		cout << "GPU clock rate: " << clockRate << " MHz" << "\n";
		cout << "Theoretical peak performance: " << GFLOPS << " GFLOPs\n" << endl;
	}
	return GFLOPS;
}
__global__ void zeroInt2DKernel(float *I, unsigned int Nq, unsigned int Nfi){
	//zeroing the 2d intensity array 
	unsigned int iq = blockDim.y*blockIdx.y + threadIdx.y, ifi = blockDim.x*blockIdx.x + threadIdx.x;
	if ((iq < Nq) && (ifi < Nfi))	I[iq*Nfi + ifi] = 0;
}
__global__ void zeroAmp2DKernel(float *Ar, float *Ai, unsigned int Nq, unsigned int Nfi){
	//zeroint real and imaginary parts of the 2d amplitude array 
	unsigned int iq = blockDim.y*blockIdx.y + threadIdx.y, ifi = blockDim.x*blockIdx.x + threadIdx.x;
	if ((iq < Nq) && (ifi < Nfi)){
		Ar[iq*Nfi + ifi] = 0;
		Ai[iq*Nfi + ifi] = 0;
	}
}
__global__ void Sum2DKernel(float *I,const float *Ar,const float *Ai, unsigned int Nq, unsigned int Nfi){
	//calculating the 2d intensity array from real and imaginary parts of the 2d amplitude array
	unsigned int iq = blockDim.y*blockIdx.y + threadIdx.y, ifi = blockDim.x*blockIdx.x + threadIdx.x;
	if ((iq < Nq) && (ifi < Nfi))	I[iq*Nfi + ifi] += SQR(Ar[iq*Nfi + ifi]) + SQR(Ai[iq*Nfi + ifi]);
}
__global__ void Norm2DKernel(float *I, unsigned int Nq, unsigned int Nfi, float norm){
	//multiplying the 2d intensity array by a normalizing factor
	unsigned int iq = blockDim.y*blockIdx.y + threadIdx.y, ifi = blockDim.x*blockIdx.x + threadIdx.x;
	if ((iq < Nq) && (ifi < Nfi))	I[iq*Nfi + ifi] *= norm;
}
template <unsigned int BlockSize2D> __global__ void PolarFactor2DKernel(float *I, unsigned int Nq, unsigned int Nfi, const float *q, float lambda){
	//computing the polarizing factor and mutiplying the 2d intensity array by it
	unsigned int iq = BlockSize2D*blockIdx.y + threadIdx.y, ifi = BlockSize2D*blockIdx.x + threadIdx.x;
	unsigned int iqCopy = BlockSize2D*blockIdx.y + threadIdx.x;
	__shared__ float factor[BlockSize2D];
	if ((threadIdx.y == 0) && (iqCopy < Nq)) {
		//polarizing factor is computed only by the threads of the first warp (half-warp for the devices with CC < 2.0) and stored in the shared memory
		float sintheta = q[iqCopy] * (lambda * 0.25f / PIf);
		float cos2theta = 1.f - 2.f * SQR(sintheta);
		factor[threadIdx.x] = 0.5f * (1.f + SQR(cos2theta));
	}
	__syncthreads();
	if ((iq < Nq) && (ifi < Nfi)) I[iq*Nfi + ifi] *= factor[threadIdx.y]; 
}
__global__ void PolarFactor1DKernel(float *I, unsigned int Nq, const float *q, float lambda){
	//computes polarizing factor and mutiplies 1d intensity array by it
	unsigned int iq = blockIdx.x*blockDim.x + threadIdx.x;
	if (iq < Nq)	{
		float sintheta = q[iq] * (lambda * 0.25f / PIf);
		float cos2theta = 1.f - 2.f * SQR(sintheta);
		float factor = 0.5f * (1.f + SQR(cos2theta));
		I[blockIdx.y * Nq + iq] *= factor;
	}
}
template <unsigned int BlockSize2D, unsigned int SizeR> __global__ void calcInt2DKernelXray(float *Ar, float *Ai, const float *q, unsigned int Nq, unsigned int Nfi, float3 CS[], float lambda, const float4 *ra, unsigned int Nfin, const float *FF){
	//this kernel computes real and imaginary parts of the 2D Xray scattering amplitude in polar coordinates (q,fi)
	//to avoid bank conflicts for shared memory operations BlockSize2D should be equal to the size of the warp (or half-warp for the devices with the CC < 2.0)
	//SizeR should be a multiple of BlockSize2D
	unsigned int iq=BlockSize2D*blockIdx.y+threadIdx.y,ifi=BlockSize2D*blockIdx.x+threadIdx.x; //each thread computes only one element of 2D amplitude matrix
	unsigned int iqCopy = BlockSize2D*blockIdx.y + threadIdx.x;//copying of the scattering vector modulus array to the shared memory performed by the threads of the same warp (half-warp)
	__shared__ float lFF[BlockSize2D]; //cache array for the atomic from-factors
	__shared__ float qi[BlockSize2D]; //cache array for the scattering vector modulus
	__shared__ float4 r[SizeR]; //cache array for the atomic coordinates
	unsigned int Niter = Nfin / SizeR + BOOL(Nfin % SizeR);//we don't have enough shared memory to load the array of atomic coordinates as a whole, so we do it with iterations
	float3 qv; //scattering vector
	float lAr = 0, lAi = 0, cosfi = 0, sinfi = 0, sintheta = 0, costheta = 0;
	if ((threadIdx.y == 0) && (iqCopy < Nq)) lFF[threadIdx.x] = FF[iqCopy]; //loading Xray form-factors to the shared memory (only threads from the first warp (half-warp) are used)
	if ((threadIdx.y == 2) && (iqCopy < Nq)) qi[threadIdx.x] = q[iqCopy]; //loading scattering vector modulus to shared memory (only threads from the third warp (first half of the second warp) are used)
	__syncthreads(); //synchronizing after loading to the shared memory
	if ((iq < Nq) && (ifi < Nfi)){//checking the 2d array margins
		__sincosf(ifi*2.f*PIf / Nfi, &sinfi, &cosfi); //computing sin(fi), cos(fi)
		sintheta = 0.25f*lambda*qi[threadIdx.y] / PIf; //q = 4pi/lambda*sin(theta)
		costheta = 1.f - SQR(sintheta); //theta in [0, pi/2];
		qv = make_float3(costheta*cosfi, costheta*sinfi, -sintheta)*qi[threadIdx.y]; //computing the scattering vector
		//instead of pre-multiplying the atomic coordinates by the rotational matrix we are pre-multiplying the scattering vector by the transposed rotational matrix (dot(qv,r) will be the same)
		qv = make_float3(dot(qv, CS[0]), dot(qv, CS[1]), dot(qv, CS[2]));
	}
	for (unsigned int iter = 0; iter < Niter; iter++){
		unsigned int NiterFin = MIN(Nfin - iter * SizeR, SizeR); //checking for the margins of the atomic coordinates array
		if (threadIdx.y < SizeR / BlockSize2D) {
			unsigned int iAtom = threadIdx.y*BlockSize2D + threadIdx.x; 
			if (iAtom < NiterFin) r[iAtom] = ra[iter * SizeR + iAtom]; //loading the atomic coordinates to the shared memory
		}
		__syncthreads(); //synchronizing after loading to shared memory
		if ((iq < Nq) && (ifi < Nfi)){//checking the 2d array margins
			for (unsigned int iAtom = 0; iAtom < NiterFin; iAtom++){
				__sincosf(dot(qv, r[iAtom]), &sinfi, &cosfi); //cos(dot(qv*r)), sin(dot(qv,r))
				lAr += cosfi; //real part of the amplitute
				lAi += sinfi; //imaginary part of the amplitute
			}
		}
		__syncthreads(); //synchronizing before the next loading starts
	}
	if ((iq < Nq) && (ifi < Nfi)){//checking the 2d array margins
		Ar[iq*Nfi + ifi] += lFF[threadIdx.y] * lAr; //multiplying the real part of the amplitude by the form-factor and writing the results to the global memory
		Ai[iq*Nfi + ifi] += lFF[threadIdx.y] * lAi; //doing the same for the imaginary part of the amplitude
	}	
}
template <unsigned int BlockSize2D, unsigned int SizeR> __global__ void calcInt2DKernelNeutron(float *Ar, float *Ai, const float *q, unsigned int Nq, unsigned int Nfi, float3 CS[], float lambda, const float4 *ra, unsigned int Nfin, float SL){
	//this kernel computes real and imaginary parts of the 2D neutron scattering amplitude in polar coordinates (q,fi)
	//see comments in the calcInt2DKernelXray() kernel
	unsigned int iq = BlockSize2D*blockIdx.y + threadIdx.y, ifi = BlockSize2D*blockIdx.x + threadIdx.x; 
	unsigned int iqCopy = BlockSize2D*blockIdx.y + threadIdx.x;
	__shared__ float qi[BlockSize2D]; 
	__shared__ float4 r[SizeR];	
	unsigned int Niter = Nfin / SizeR + BOOL(Nfin % SizeR);
	float3 qv;
	float lAr = 0, lAi = 0, cosfi = 0, sinfi = 0, sintheta = 0, costheta = 0;
	if ((threadIdx.y == 0) && (iqCopy < Nq)) qi[threadIdx.x] = q[iqCopy];
	__syncthreads();
	if ((iq < Nq) && (ifi < Nfi)){
		__sincosf(ifi*2.f*PIf / Nfi, &sinfi, &cosfi);
		sintheta = 0.25f*lambda*qi[threadIdx.y] / PIf;
		costheta = 1.f - SQR(sintheta);
		qv = make_float3(costheta*cosfi, costheta*sinfi, -sintheta)*qi[threadIdx.y];
		qv = make_float3(dot(qv, CS[0]), dot(qv, CS[1]), dot(qv, CS[2]));
	}
	for (unsigned int iter = 0; iter < Niter; iter++){
		unsigned int NiterFin = MIN(Nfin - iter * SizeR, SizeR);
		if (threadIdx.y < SizeR / BlockSize2D) {
			unsigned int iAtom = threadIdx.y*BlockSize2D + threadIdx.x; 
			if (iAtom < NiterFin) r[iAtom] = ra[iter * SizeR + iAtom];
		}
		__syncthreads();
		if ((iq < Nq) && (ifi < Nfi)){
			for (unsigned int iAtom = 0; iAtom < NiterFin; iAtom++){
				__sincosf(dot(qv, r[iAtom]), &sinfi, &cosfi);
				lAr += cosfi;
				lAi += sinfi;
			}
		}
		__syncthreads();
	}
	if ((iq < Nq) && (ifi < Nfi)){
		Ar[iq*Nfi + ifi] += SL*lAr;
		Ai[iq*Nfi + ifi] += SL*lAi;
	}
}

void calcInt2DCuda(unsigned int DeviceNUM, double ***I2D, double **I, const calc *cfg, const unsigned int *NatomType, const float4 *ra, const float * const *dFF, vector<double> SL, const float *dq){
	//computing 2d scattering intensity in polar coordinates (q,fi)
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	unsigned int MaxAtomsPerLaunch = 0, BlockSize2D = BlockSize2Dsmall, Ntot = 0;
	float *hI, *dI, *dAr, *dAi;
	*I = new double[cfg->q.N]; //array for 1d scattering intensity I[q] (I2D[q][fi] averaged over polar angle fi)
	*I2D = new double*[cfg->q.N]; //array for 2d scattering intensity 
	for (unsigned int iq = 0; iq<cfg->q.N; iq++){
		(*I)[iq] = 0;
		(*I2D)[iq] = new double[cfg->Nfi];
	}
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, DeviceNUM); //getting device information
	unsigned int GFLOPS = GetGFLOPS(deviceProp); //theoretical peak GPU performance
	if (deviceProp.kernelExecTimeoutEnabled){ //killswitch is enabled, so the time limit should not be exceeded
		const double tmax = 0.2; //maximum kernel execution time in seconds
		const double k = 4.e-8; // t = k * MaxAtomsPerLaunch * Nq * Nfi / GFLOPS
		MaxAtomsPerLaunch = (unsigned int)((tmax*GFLOPS) / (k*cfg->q.N*cfg->Nfi)); //maximum number of atoms per kernel launch
	}
	for (unsigned int iType = 0; iType<cfg->Ntype; iType++) Ntot += NatomType[iType]; //total number of atoms
	unsigned int Nm = cfg->q.N*cfg->Nfi; //dimension of 2D intensity array
	hI = new float[Nm]; //host array for 2D intensity
	//allocating memory on the device for amplitude and intensity 2D arrays
	//GPU has linear memory, so we stretch 2D arrays into 1D arrays
	cudaMalloc(&dAr, Nm*sizeof(float));
	cudaMalloc(&dAi, Nm*sizeof(float));
	cudaMalloc(&dI, Nm*sizeof(float));	
	cudaThreadSynchronize(); //synchronizing before calculating the amplitude
	dim3 dimBlock(BlockSize2D, BlockSize2D); //2d thread block size
	dim3 dimGrid(cfg->Nfi / BlockSize2D + BOOL(cfg->Nfi % BlockSize2D), cfg->q.N / BlockSize2D + BOOL(cfg->q.N % BlockSize2D)); //grid size
	float3 CS[3], *dCS; //three rows of the transposed rotational matrix for the host and the device
	unsigned int Nst, Nfin;
	//2d scattering intensity should be calculated for the preset orientation of the sample (or averaged over multiple orientations specified by mesh)
	double dalpha = (cfg->Euler.max.x - cfg->Euler.min.x) / cfg->Euler.N.x, dbeta = (cfg->Euler.max.y - cfg->Euler.min.y) / cfg->Euler.N.y, dgamma = (cfg->Euler.max.z - cfg->Euler.min.z) / cfg->Euler.N.z;
	if (cfg->Euler.N.x<2) dalpha = 0;
	if (cfg->Euler.N.y<2) dbeta = 0;
	if (cfg->Euler.N.z<2) dgamma = 0;
	cudaMalloc(&dCS, 3 * sizeof(float3)); //allocating the device memory for the transposed rotational matrix
	zeroInt2DKernel << <dimGrid, dimBlock >> >(dI, cfg->q.N, cfg->Nfi); //zeroing the 2D intensity matrix
	vect3d <double> cf0, cf1, cf2; //three rows of the rotational matrix
	for (unsigned int ia = 0; ia < cfg->Euler.N.x; ia++){
		double alpha = cfg->Euler.min.x + (ia + 0.5)*dalpha;
		for (unsigned int ib = 0; ib < cfg->Euler.N.y; ib++){
			double beta = cfg->Euler.min.y + (ib + 0.5)*dbeta;
			for (unsigned int ig = 0; ig < cfg->Euler.N.z; ig++){
				double gamma = cfg->Euler.min.z + (ig + 0.5)*dgamma;
				vect3d <double> euler(alpha, beta, gamma);
				calcRotMatrix(&cf0, &cf1, &cf2, euler, cfg->EulerConvention); //calculating the rotational matrix
				CS[0]=make_float3(float(cf0.x), float(cf1.x), float(cf2.x)); //transposing the rotational matrix
				CS[1]=make_float3(float(cf0.y), float(cf1.y), float(cf2.y));
				CS[2]=make_float3(float(cf0.z), float(cf1.z), float(cf2.z));
				cudaMemcpy(dCS, CS, 3 * sizeof(float3), cudaMemcpyHostToDevice); //copying transposed rotational matrix from the host memory to the device memory 
				zeroAmp2DKernel << <dimGrid, dimBlock >> >(dAr, dAi, cfg->q.N, cfg->Nfi); //zeroing 2D amplitude arrays
				cudaThreadSynchronize(); //synchronizing before calculation starts to ensure that amplitude arrays were successfully set to zero
				unsigned int inp = 0;
				for (unsigned int iType = 0; iType < cfg->Ntype; iType++){ //looping over chemical elements (or ions)
					if (MaxAtomsPerLaunch) { //killswitch is enabled so MaxAtomsPerLaunch is set
						for (unsigned int i = 0; i < NatomType[iType] / MaxAtomsPerLaunch + BOOL(NatomType[iType] % MaxAtomsPerLaunch); i++) { //looping over the iterations
							Nst = inp + i*MaxAtomsPerLaunch; //index for the first atom on the current iteration step
							Nfin = MIN(Nst + MaxAtomsPerLaunch, inp + NatomType[iType]) - Nst; //index for the last atom on the current iteration step
							//float time; //time control sequence
							//cudaEvent_t start, stop;
							//cudaEventCreate(&start);
							//cudaEventCreate(&stop);
							//cudaEventRecord(start, 0);
							if (cfg->source) { //Xray scattering 
								calcInt2DKernelXray <BlockSize2Dsmall, 8 * BlockSize2Dsmall> << <dimGrid, dimBlock >> >(dAr, dAi, dq, cfg->q.N, cfg->Nfi, dCS, float(cfg->lambda), ra + Nst, Nfin, dFF[iType]);
							}
							else {//neutron scattering
								calcInt2DKernelNeutron <BlockSize2Dsmall, 8 * BlockSize2Dsmall> << <dimGrid, dimBlock >> >(dAr, dAi, dq, cfg->q.N, cfg->Nfi, dCS, float(cfg->lambda), ra + Nst, Nfin, float(SL[iType]));
							}
							cudaThreadSynchronize(); //synchronizing to ensure that additive operations does not overlap
							//cudaEventRecord(stop, 0);
							//cudaEventSynchronize(stop);
							//cudaEventElapsedTime(&time, start, stop);
							//cout << "calcInt2DKernel execution time is: " << time << " ms\n" << endl;
						}
					}
					else { //killswitch is disabled so we execute the kernels for the entire ensemble of atoms
						Nst = inp;
						Nfin = NatomType[iType];
						if (cfg->source) { //Xray scattering 
							calcInt2DKernelXray <BlockSize2Dsmall, 8 * BlockSize2Dsmall> << <dimGrid, dimBlock >> >(dAr, dAi, dq, cfg->q.N, cfg->Nfi, dCS, float(cfg->lambda), ra + Nst, Nfin, dFF[iType]);
						}
						else {//neutron scattering
							calcInt2DKernelNeutron <BlockSize2Dsmall, 8 * BlockSize2Dsmall> << <dimGrid, dimBlock >> >(dAr, dAi, dq, cfg->q.N, cfg->Nfi, dCS, float(cfg->lambda), ra + Nst, Nfin, float(SL[iType]));
						}
						cudaThreadSynchronize(); //synchronizing to ensure that additive operations does not overlap
					}
					inp += NatomType[iType];
				}				
				Sum2DKernel << <dimGrid, dimBlock >> >(dI, dAr, dAi, cfg->q.N, cfg->Nfi); //calculating the 2d scattering intensity by the scattering amplitude
			}
		}
	}
	float norm = 1.f / (Ntot*cfg->Euler.N.x*cfg->Euler.N.y*cfg->Euler.N.z); //normalizing factor
	Norm2DKernel << <dimGrid, dimBlock >> >(dI, cfg->q.N, cfg->Nfi, norm); //normalizing the 2d scattering intensity
	cudaThreadSynchronize(); //synchronizing to ensure that multiplying operations does not overlap
	if (cfg->PolarFactor) { //multiplying the 2d intensity by polar factor
		PolarFactor2DKernel <BlockSize2Dsmall> << <dimGrid, dimBlock >> >(dI, cfg->q.N, cfg->Nfi, dq, float(cfg->lambda));
	}
	cudaMemcpy(hI, dI, Nm*sizeof(float), cudaMemcpyDeviceToHost);  //copying the 2d intensity matrix from the device memory to the host memory 
	for (unsigned int iq = 0; iq < cfg->q.N; iq++){
		for (unsigned int ifi = 0; ifi < cfg->Nfi; ifi++)	{
			(*I2D)[iq][ifi] = double(hI[iq*cfg->Nfi + ifi]);
			(*I)[iq] += (*I2D)[iq][ifi]; //calculating the 1d intensity (averaging I2D[q][fi] over the polar angle fi)
		}
		(*I)[iq] /= cfg->Nfi;
	}
	//deallocating the device memory
	cudaFree(dCS);
	cudaFree(dAr);
	cudaFree(dAi);
	cudaFree(dI);
	//deallocating the host memory
	delete[] hI;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cout << "2D pattern calculation time: " << time/1000 << " s" << endl;
}
__global__ void zero1DFloatArrayKernel(float *A, unsigned int N){
	//zeroing any 1d float array
	unsigned int i=blockDim.x*blockIdx.x+threadIdx.x;
	if (i<N) A[i]=0;
}
__global__ void addIKernelXray(float *I, const float *FFi, unsigned int Nq, unsigned int N) {
	//adding the diagonal elements to the Xray scattering intensity (j==i in the Debye double sum) 
	unsigned int iq = blockIdx.x*blockDim.x + threadIdx.x;
	if (iq < Nq)	{
		float lFF = FFi[iq];
		I[iq] += SQR(lFF) * N;
	}
}
__global__ void addIKernelNeutron(float *I, unsigned int Nq, float Add) {
	//adding the diagonal elements to the neutron scattering intensity (j==i in the Debye double sum) 
	unsigned int iq = blockIdx.x*blockDim.x + threadIdx.x;
	if (iq < Nq)	I[iq] += Add;
}
__global__ void sumIKernel(float *I, unsigned int Nq, unsigned int Nsum){
	//reducing the grid array of scattering intensity into final array (first Nq elements)
	unsigned int iq = blockDim.x * blockIdx.x + threadIdx.x;
	if (iq<Nq) {
		for (unsigned int j = 1; j < Nsum; j++)	I[iq] += I[j*Nq+iq];
	}
}
__global__ void zeroHistKernel(unsigned long long int *rij_hist,unsigned int N){
	//zeroing the pair distribution histogram
	unsigned int i=blockDim.x*blockIdx.x+threadIdx.x;
	if (i<N) rij_hist[i]=0;
}	
template <unsigned int BlockSize2D> __global__ void calcHistKernel(const float4 *ri,const float4 *rj, unsigned int iMax, unsigned int jMax, unsigned long long int *rij_hist, float bin, unsigned int Nhistcopies, unsigned int Nhist, bool diag){
	//this kernel calculates the pair-distribution histogram
	if ((diag) && (blockIdx.x < blockIdx.y)) return; //we need to calculate inter-atomic distances only for j > i, so if we are in the diagonal grid, all the subdiagonal blocks (for which j < i for all threads) do nothing and return
	unsigned int jt = threadIdx.x, it = threadIdx.y;
	unsigned int j = blockIdx.x * BlockSize2D + jt;
	unsigned int iCopy = blockIdx.y * BlockSize2D + jt; //jt!!! memory transaction are performed by the threads of the same warp to coalesce them
	unsigned int i = blockIdx.y * BlockSize2D + it;
	unsigned int copyind = 0;
	if (Nhistcopies>1) copyind = ((it * BlockSize2D + jt) % Nhistcopies)*Nhist; //some optimization for CC < 2.0. Making multiple copies of the histogram array reduces the number of atomicAdd() operations on the same elements.
	__shared__ float4 ris[BlockSize2D], rjs[BlockSize2D]; //cache arrays for atomic coordinates (we use float3 here to avoid bank conflicts)
	if ((it == 0) && (j < jMax)) { //copying atomic coordinates for j-th (column) atoms (only the threads of the first half-warp are used)
		rjs[jt] = rj[j];
	}
	if ((it == 2) && (iCopy < iMax)) { //the same for i-th (row) atoms (only the threads of the first half-warp of the second warp for CC < 2.0 are used)
		ris[jt] = ri[iCopy];
	}
	__syncthreads(); //sync to ensure that copying is complete
	if (!diag){
		if ((j < jMax) && (i < iMax)) {
			float rij = sqrtf(SQR(ris[it].x - rjs[jt].x) + SQR(ris[it].y - rjs[jt].y) + SQR(ris[it].z - rjs[jt].z));//calculate distance
			unsigned int index = (unsigned int)(rij / bin); //get the index of histogram bin
			atomicAdd(&rij_hist[copyind + index], 1); //add +1 to histogram bin
		}
	}
	else{//we are in diagonal grid
		if ((j < jMax) && (i < iMax) && (j > i)) {//all the subdiagonal blocks already quit, but we have diagonal blocks  (blockIdx.x == blockIdx.y), so we should check if j > i
			float rij = sqrtf(SQR(ris[it].x - rjs[jt].x) + SQR(ris[it].y - rjs[jt].y) + SQR(ris[it].z - rjs[jt].z));
			unsigned int index = (unsigned int)(rij / bin);
			atomicAdd(&rij_hist[copyind + index], 1);
		}
	}
}
__global__ void sumHistKernel(unsigned long long int *rij_hist, unsigned  int Nhistcopies, unsigned int Nfin, unsigned int Nhist){
	//This is for CC < 2.0 only. This kernel reduces the copies of pair-distribution histogram into final array (first Nhist elements)
	unsigned int i=blockDim.x*blockIdx.x+threadIdx.x;
	if (i<Nfin){
		for (unsigned int iCopy = 1; iCopy<Nhistcopies; iCopy++)	rij_hist[i] += rij_hist[Nhist*iCopy + i];
	}
}
void calcHistCuda(unsigned int DeviceNUM, unsigned long long int **rij_hist, const float4 *ra, const unsigned int *NatomType, unsigned int Ntype, unsigned int Nhist, float bin){
	//this function calculates the pair-distribution histogram
	unsigned int GridSizeExecMax = 2048;
	unsigned int BlockSize = BlockSize1Dsmall, BlockSize2D = BlockSize2Dsmall; //size of the thread blocks (256, 16x16)
	unsigned int Nhistcopies = 1, NhistType = (Ntype*(Ntype + 1)) / 2 * Nhist;//NhistType - number of partial (Element1<-->Element2) histograms
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, DeviceNUM); //getting the device properties
	int cc = deviceProp.major * 10 + deviceProp.minor; //device compute capability
	if (cc<20){//optimization for the devices with CC < 2.0
		//atomic operations work very slow for the devices with Tesla architecture as compared with the modern devices
		//we minimize the number of atomic operations on the same elements by making multiple copies of pair-distribution histograms
		size_t free, total;
		cuMemGetInfo(&free, &total); //checking the amount of the free GPU memory	
		Nhistcopies = MIN(BlockSize,(unsigned int)(0.25*float(free) / (NhistType*sizeof(unsigned long long int)))); //set optimal number for histogram copies 
		if (!Nhistcopies) Nhistcopies = 1;
	}
	unsigned int GFLOPS = GetGFLOPS(deviceProp); //theoretical peak GPU performance
	if (deviceProp.kernelExecTimeoutEnabled)	{//killswitch is enabled, so the time limit should not be exceeded
		const double tmax = 0.1; //maximum kernel time execution in seconds
		const double k = 1.e-6; // t = k * GridSizeExecMax^2 * BlockSize2D^2 / GFLOPS
		GridSizeExecMax = MIN((unsigned int)(sqrt(tmax * GFLOPS / k) / BlockSize2D), GridSizeExecMax);
	}
	//total histogram size is equal to the product of: partial histogram size for one pair of elements (Nhist), number of partial histograms ((Ntype*(Ntype + 1)) / 2), number of histogram copies (Nhistcopies)
	unsigned int NhistTotal = NhistType * Nhistcopies;
	cudaError err = cudaMalloc(rij_hist, NhistTotal*sizeof(unsigned long long int));//trying to allocate large amount of memory, check for errors
	if (err != cudaSuccess) cout << "Error in calcHistCuda(), cudaMalloc(): " << cudaGetErrorString(err) << endl;
	unsigned int GSzero = MIN(65535, NhistTotal / BlockSize + BOOL(NhistTotal % BlockSize));//Size of the grid for zeroHistKernel (it could not be large than 65535)
	//zeroing pair-distribution histogram array
	for (unsigned int iter = 0; iter < NhistTotal / BlockSize + BOOL(NhistTotal % BlockSize); iter += GSzero)	zeroHistKernel << < GSzero, BlockSize >> >(*rij_hist + iter*BlockSize, NhistTotal - iter*BlockSize);
	cudaThreadSynchronize();//synchronizing before the calculation starts
	dim3 blockgrid(BlockSize2D, BlockSize2D);//2D thread block size
	unsigned int Nstart = 0, jAtom0, iAtomST = 0;
	bool diag = false;
	for (unsigned int iType = 0; iType < Ntype; iAtomST += NatomType[iType], iType++) {
		unsigned int jAtomST = iAtomST;
		for (unsigned int jType = iType; jType < Ntype; jAtomST += NatomType[jType], jType++, Nstart += Nhist) {//each time we move to the next pair of elements (iType,jType) we also move to the respective part of histogram (Nstart += Nhist)
			for (unsigned int iAtom = 0; iAtom < NatomType[iType]; iAtom += BlockSize2D*GridSizeExecMax){
				unsigned int GridSizeExecY = MIN((NatomType[iType] - iAtom) / BlockSize2D + BOOL((NatomType[iType] - iAtom) % BlockSize2D), GridSizeExecMax);//Y-size of the grid on the current step
				unsigned int iMax = MIN(BlockSize2D*GridSizeExecY, NatomType[iType] - iAtom);//index of the last i-th (row) atom
				(iType == jType) ? jAtom0 = iAtom : jAtom0 = 0;//loop should exclude subdiagonal grids
				for (unsigned int jAtom = jAtom0; jAtom < NatomType[jType]; jAtom += BlockSize2D*GridSizeExecMax){
					unsigned int GridSizeExecX = MIN((NatomType[jType] - jAtom) / BlockSize2D + BOOL((NatomType[jType] - jAtom) % BlockSize2D), GridSizeExecMax);//X-size of the grid on the current step
					unsigned int jMax = MIN(BlockSize2D*GridSizeExecX, NatomType[jType] - jAtom);//index of the last j-th (column) atom
					dim3 grid(GridSizeExecX, GridSizeExecY);
					(iAtomST + iAtom == jAtomST + jAtom) ? diag = true : diag = false;//checking if we are on the diagonal grid or not
					/*float time;
					cudaEvent_t start, stop;
					cudaEventCreate(&start);
					cudaEventCreate(&stop);
					cudaEventRecord(start, 0);*/
					calcHistKernel <BlockSize2Dsmall> << <grid, blockgrid >> >(ra + iAtomST + iAtom, ra + jAtomST + jAtom, iMax, jMax, *rij_hist + Nstart, bin, Nhistcopies, NhistType, diag);
					if (deviceProp.kernelExecTimeoutEnabled) cudaThreadSynchronize();//the kernels above use atomic operation, and it's impossible to predict the execution time of a single kernell call, so sync to avoid the killswitch triggering 
					/*cudaEventRecord(stop, 0);
					cudaEventSynchronize(stop);
					cudaEventElapsedTime(&time, start, stop);
					cout << "calcHistKernel execution time is: " << time << " ms\n" << endl;*/
				}
			}
		}
	}
	cudaThreadSynchronize();//synchronizing to ensure that all calculations ended before histogram copies summation starts
	if (Nhistcopies>1) {//summing the histogram copies
		unsigned int GSsum = MIN(65535, NhistType / BlockSize + BOOL(NhistType % BlockSize));
		for (unsigned int iter = 0; iter < NhistType / BlockSize + BOOL(NhistType % BlockSize); iter += GSsum)	sumHistKernel << <GSsum, BlockSize >> >(*rij_hist + iter*BlockSize, Nhistcopies, NhistType - iter*BlockSize, NhistType);
	}
	cudaThreadSynchronize();//synchronizing before the further usage of histogram in other functions
}
template <unsigned int Size>__global__ void calcIntHistKernelXray(float *I,const float *FFi,const float *FFj,const float *q, unsigned int Nq,const unsigned long long int *rij_hist, unsigned int Nhist, unsigned int MaxBinsPerBlock, float bin){
	//this kernel calculates the Xray scattering intensity using the pair-distribution histogram
	__shared__ long long int Nrij[Size];//cache array for the histogram
	unsigned int iBegin = blockIdx.x * MaxBinsPerBlock;//first index for histogram bin to process
	unsigned int iEnd = MIN(Nhist, iBegin + MaxBinsPerBlock);//last index for histogram bin to process
	unsigned int Niter = (iEnd - iBegin) / blockDim.x + BOOL((iEnd - iBegin) % blockDim.x);//number of iterations
	for (unsigned int iter = 0; iter < Niter; iter++){//we don't have enough shared memory to load histogram array as a whole, so we do it with iterations
		unsigned int NiterFin = MIN(iEnd - iBegin - iter * blockDim.x, blockDim.x);//maximum number of histogram bins on current iteration step
		if (threadIdx.x < NiterFin) Nrij[threadIdx.x] = rij_hist[iBegin + iter * blockDim.x + threadIdx.x];//loading the histogram array to the shared memory
		__syncthreads();//synchronizing after loading
		for (unsigned int iterq = 0; iterq < (Nq / blockDim.x) + BOOL(Nq % blockDim.x); iterq++) {//if Nq > blockDim.x there will be threads that compute more than one element of the intensity array
			unsigned int iq = iterq*blockDim.x + threadIdx.x;//index of the intensity array element
			if (iq < Nq) {//checking for the array margin
				float lI=0, qrij;
				float lq = q[iq];//copying the scattering vector modulus to the local memory
				for (unsigned int i = 0; i < NiterFin; i++) {//looping over the histogram bins
					if (Nrij[i]){
						qrij = lq * ((float)(iBegin + iter * blockDim.x + i) + 0.5f)*bin;//distance that corresponds to the current histogram bin
						lI += (Nrij[i] * __sinf(qrij)) / (qrij + 0.000001f);//scattering intensity without form factors
					}
				}
				float lFFij = 2.f * FFi[iq] * FFj[iq];
				I[blockIdx.x*Nq + iq] += lI * lFFij;//multiplying intensity by form-factors and storing the results in global memory
			}
		}
		__syncthreads();//synchronizing threads before the next iteration step
	}
}
template <unsigned int Size>__global__ void calcIntHistKernelNeutron(float *I, float SLij,const float *q, unsigned int Nq,const unsigned long long int *rij_hist, unsigned int Nhist, unsigned int MaxBinsPerBlock, float bin){
	//this kernel calculates neutron scattering intensity using pair distribution histogram
	//see comments in the calcIntHistKernelXray() kernel
	__shared__ long long int Nrij[Size];
	unsigned int iBegin = blockIdx.x * MaxBinsPerBlock;
	unsigned int iEnd = MIN(Nhist, iBegin + MaxBinsPerBlock);
	unsigned int Niter = (iEnd - iBegin) / blockDim.x + BOOL((iEnd - iBegin) % blockDim.x);
	unsigned int Nqiter = (Nq / blockDim.x) + BOOL(Nq % blockDim.x);
	for (unsigned int iter = 0; iter < Niter; iter++){
		unsigned int NiterFin = MIN(iEnd - iBegin - iter * blockDim.x, blockDim.x);
		if (threadIdx.x < NiterFin) Nrij[threadIdx.x] = rij_hist[iBegin + iter * blockDim.x + threadIdx.x];
		__syncthreads();
		for (unsigned int iterq = 0; iterq < Nqiter; iterq++) {
			unsigned int iq = iterq*blockDim.x + threadIdx.x;
			if (iq < Nq) {
				float lI = 0, qrij;
				float lq = q[iq];
				for (unsigned int i = 0; i < NiterFin; i++) {
					if (Nrij[i]){
						qrij = lq * ((float)(iBegin + iter * blockDim.x + i) + 0.5f)*bin;
						lI += (Nrij[i] * __sinf(qrij)) / (qrij + 0.000001f);
					}
				}
				I[blockIdx.x*Nq + iq] += 2.f * lI * SLij;
			}
		}
		__syncthreads();
	}
}
void calcInt1DHistCuda(unsigned int DeviceNUM, double **I, const unsigned long long int *rij_hist, const unsigned int *NatomType, const calc *cfg, const float * const * dFF, vector<double> SL, const float *dq, unsigned int Ntot){
	//calculates Xray or neutron scattering intensity using pair distribution histogram
	unsigned int BlockSize = BlockSize1Dlarge;//setting the size of the thread blocks to 1024 (default)
	float *hI = NULL, *dI = NULL;//host and device arrays for scattering intensity
	*I = new double[cfg->q.N];
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, DeviceNUM);//getting device properties
	int cc = deviceProp.major * 10 + deviceProp.minor;//device compute capability
	if (cc < 20) BlockSize = BlockSize1Dmedium;//setting the size of the thread blocks to 512 for the devices with CC < 2.0
	unsigned int GridSize = MIN(256, cfg->Nhist / BlockSize + BOOL(cfg->Nhist % BlockSize));
	unsigned int MaxBinsPerBlock = cfg->Nhist / GridSize + BOOL(cfg->Nhist % GridSize);
	unsigned int GFLOPS = GetGFLOPS(deviceProp);//theoretical peak GPU performance
	if (deviceProp.kernelExecTimeoutEnabled)	{//killswitch is enabled, so the time limit should not be exceeded
		const double tmax = 0.2; //maximum kernel time execution in seconds
		const double k = 2.5e-5; // t = k * Nq * MaxBinsPerBlock / GFLOPS
		MaxBinsPerBlock = MIN((unsigned int)(tmax*GFLOPS/(k*cfg->q.N)), MaxBinsPerBlock);
	}
	unsigned int Isize = GridSize*cfg->q.N;//each block writes to it's own copy of scattering intensity array
	cudaMalloc(&dI, Isize*sizeof(float));//allocating the device memory for the scattering intensity array
	unsigned int GSzero = MIN(65535, Isize / BlockSize + BOOL(Isize % BlockSize));//grid size for zero1DFloatArrayKernel
	for (unsigned int iter = 0; iter < Isize / BlockSize + BOOL(Isize % BlockSize); iter += GSzero) zero1DFloatArrayKernel << <GSzero, BlockSize >> >(dI + iter*BlockSize, Isize - iter*BlockSize);//zeroing intensity array
	cudaThreadSynchronize();//synchronizing before calculation starts
	unsigned int Nstart = 0, GSadd = cfg->q.N / BlockSize1Dsmall + BOOL(cfg->q.N % BlockSize1Dsmall);//grid size for addIKernelXray/addIKernelNeutron
	for (unsigned int iType = 0; iType < cfg->Ntype; iType++) {
		if (cfg->source) addIKernelXray << <GSadd, BlockSize1Dsmall >> > (dI, dFF[iType], cfg->q.N, NatomType[iType]);//add contribution form diagonal (i==j) elements in Debye sum
		else addIKernelNeutron << <GSadd, BlockSize1Dsmall >> > (dI, cfg->q.N, float(SQR(SL[iType]) * NatomType[iType]));
		cudaThreadSynchronize();//synchronizing before main calculation starts
		for (unsigned int jType = iType; jType < cfg->Ntype; jType++, Nstart += cfg->Nhist){
			for (unsigned int iBin = 0; iBin < cfg->Nhist; iBin += GridSize*MaxBinsPerBlock) {//iterations to avoid killswitch triggering
				/*float time;
				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);
				cudaEventRecord(start, 0);*/
				if (cfg->source) {//Xray
					if (cc>20) calcIntHistKernelXray <BlockSize1Dlarge> <<<GridSize, BlockSize>>> (dI, dFF[iType], dFF[jType], dq, cfg->q.N, rij_hist + Nstart + iBin, cfg->Nhist, MaxBinsPerBlock, float(cfg->hist_bin));
					else calcIntHistKernelXray <BlockSize1Dmedium> <<<GridSize, BlockSize>>> (dI, dFF[iType], dFF[jType], dq, cfg->q.N, rij_hist + Nstart + iBin, cfg->Nhist, MaxBinsPerBlock, float(cfg->hist_bin));
				}
				else {//neutron
					if (cc>20) calcIntHistKernelNeutron <BlockSize1Dlarge> <<<GridSize, BlockSize>>> (dI, float(SL[iType] * SL[jType]), dq, cfg->q.N, rij_hist + Nstart + iBin, cfg->Nhist, MaxBinsPerBlock, float(cfg->hist_bin));
					else calcIntHistKernelNeutron <BlockSize1Dmedium> <<<GridSize, BlockSize>>> (dI, float(SL[iType] * SL[jType]), dq, cfg->q.N, rij_hist + Nstart + iBin, cfg->Nhist, MaxBinsPerBlock, float(cfg->hist_bin));
				}
				/*cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&time, start, stop);
				cout << "calcIntHistKernel execution time is: " << time << " ms\n" << endl;*/
				cudaThreadSynchronize();//synchronizing before the next iteration step
			}
		}
	}
	sumIKernel << <GSadd, BlockSize1Dsmall >> >(dI, cfg->q.N, GridSize);//summing intensity copies
	cudaThreadSynchronize();//synchronizing threads before multiplying the intensity by a polarization factor
	if (cfg->PolarFactor) PolarFactor1DKernel << <GSadd, BlockSize1Dsmall >> >(dI, cfg->q.N, dq, float(cfg->lambda));
	hI = new float[cfg->q.N];
	cudaMemcpy(hI, dI, cfg->q.N*sizeof(float), cudaMemcpyDeviceToHost);//copying intensity array from the device to the host
	cudaFree(dI);//deallocating memory for intensity array
	for (unsigned int iq = 0; iq<cfg->q.N; iq++) (*I)[iq] = double(hI[iq])/Ntot;//normalizing
	delete[] hI;
}
__global__ void calcPartialRDFkernel(float *dPDF, const unsigned long long int *rij_hist, unsigned int Nhist, float mult) {
	//calculating the partial radial distribution function (RDF) using the pair distribution histogram
	unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < Nhist) dPDF[i] = rij_hist[i] * mult;
}
__global__ void calcPartialPDFkernel(float *dPDF, const unsigned long long int *rij_hist, unsigned int Nhist, float mult, float bin) {
	//calculating the partial pair distribution function (PDF) using the pair distribution histogram
	unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < Nhist) {
		float r = (i + 0.5f)*bin;
		dPDF[i] = rij_hist[i] * (mult / SQR(r));
	}
}
__global__ void calcPartialRPDFkernel(float *dPDF,const unsigned long long int *rij_hist, unsigned int Nhist, float mult, float submult, float bin) {
	//calculating the partial reduced pair-distribution function (rPDF) using the pair distribution histogram
	unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < Nhist) {
		float r = (i + 0.5f)*bin;
		dPDF[i] = rij_hist[i] * (mult / r) - submult * r;
	}
}
__global__ void calcPDFkernel (float *dPDF,unsigned int Nstart,unsigned int Nhist,float multIJ) {
	//calculating the full PDF by summing the partial PDFs
	unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < Nhist) 	dPDF[i] += dPDF[Nstart + i] * multIJ;
}
void calcPDFandDebyeCuda(unsigned int DeviceNUM, double **I, double **PDF, const calc *cfg, const unsigned int *NatomType, const float4 *ra, const float * const * dFF, vector<double> SL, const float *dq) {
	//calculating the pair-distribution functions (PDFs) and/or scattering intensity using the pair-distribution histogram
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	unsigned long long int *rij_hist = NULL;//array for pair-distribution histogram (device only)
	calcHistCuda(DeviceNUM, &rij_hist, ra, NatomType, cfg->Ntype, cfg->Nhist, float(cfg->hist_bin));//calculating the histogram
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cout << "Histogramm calculation time: " << time/1000 << " s" << endl;
	unsigned int Ntot = 0;
	for (unsigned int iType = 0; iType < cfg->Ntype; iType++) Ntot += NatomType[iType];//calculating the total number of atoms
	if (cfg->scenario > 2) {//calculating the PDFs
		cudaEventRecord(start, 0);
		unsigned int BlockSize = BlockSize1Dmedium;
		unsigned int NPDF = (1 + (cfg->Ntype*(cfg->Ntype + 1)) / 2) * cfg->Nhist, NPDFh = NPDF;//total PDF array size (full (cfg->Nhist) + partial (cfg->Nhist*(cfg->Ntype*(cfg->Ntype + 1)) / 2) )
		if (!cfg->PrintPartialPDF) NPDFh = cfg->Nhist;//if the partial PDFs are not needed, we are not copying them to the host
		*PDF = new double[NPDFh];//resulting array of doubles for PDF
		float *hPDF = NULL, *dPDF = NULL;
		cudaMalloc(&dPDF, NPDF*sizeof(float));//allocating the device memory for PDF array
		float Faverage2 = 0;
		for (unsigned int iType = 0; iType < cfg->Ntype; iType++) {
			Faverage2 += float(SL[iType] * NatomType[iType]); //calculating the average form-factor
		}
		Faverage2 /= Ntot;
		Faverage2 *= Faverage2;//and squaring it
		//the size of the histogram array may exceed the maximum number of thread blocks in the grid (65535 for the devices with CC < 3.0) multiplied by the thread block size (512 for devices with CC < 2.0 or 1024 for others)
		//so any operations on histogram array should be performed iteratively
		unsigned int GSzero = MIN(65535, NPDF / BlockSize + BOOL(NPDF % BlockSize));//grid size for zero1DFloatArrayKernel
		for (unsigned int iter = 0; iter < NPDF / BlockSize + BOOL(NPDF % BlockSize); iter += GSzero)	zero1DFloatArrayKernel << <NPDF / BlockSize + BOOL(NPDF % BlockSize), BlockSize >> >(dPDF + iter*BlockSize, NPDF - iter*BlockSize);//zeroing the PDF array
		cudaThreadSynchronize();//synchronizing before calculation starts
		unsigned int Nstart = 0, GridSizeMax = (cfg->Nhist - 1) / BlockSize + BOOL((cfg->Nhist - 1) % BlockSize), GridSize = MIN(65535, GridSizeMax);//grid size for main kernels
		for (unsigned int iType = 0; iType < cfg->Ntype; iType++) {
			for (unsigned int jType = iType; jType < cfg->Ntype; jType++, Nstart += cfg->Nhist){
				float mult ,sub;
				switch (cfg->PDFtype){
					case 0://calculating partial RDFs
						mult = 2.f / (float(cfg->hist_bin)*Ntot);
						for (unsigned int iter = 0; iter < GridSizeMax; iter += GridSize)	calcPartialRDFkernel << <GridSize, BlockSize >> > (dPDF + iter*BlockSize + cfg->Nhist + Nstart, rij_hist + iter*BlockSize + Nstart, cfg->Nhist - iter*BlockSize, mult);
						break;
					case 1://calculating partial PDFs
						mult = 0.5f / (PIf*float(cfg->hist_bin*cfg->p0)*Ntot);
						for (unsigned int iter = 0; iter < GridSizeMax; iter += GridSize) calcPartialPDFkernel << <GridSize, BlockSize >> > (dPDF + iter*BlockSize + cfg->Nhist + Nstart, rij_hist + iter*BlockSize + Nstart, cfg->Nhist - iter*BlockSize, mult, float(cfg->hist_bin));
						break;
					case 2://calculating partial rPDFs
						mult = 2.f / (float(cfg->hist_bin)*Ntot);
						(jType > iType) ? sub = 8.f*PIf*float(cfg->p0)*float(NatomType[iType])* float(NatomType[jType]) / SQR(float(Ntot)) : sub=4.f*PIf*float(cfg->p0)*SQR(float(NatomType[iType])) / SQR(float(Ntot));
						for (unsigned int iter = 0; iter < GridSizeMax; iter += GridSize) calcPartialRPDFkernel << <GridSize, BlockSize >> > (dPDF + iter*BlockSize + cfg->Nhist + Nstart, rij_hist + iter*BlockSize + Nstart, cfg->Nhist - iter*BlockSize, mult, sub, float(cfg->hist_bin));
						break;
				}
			}
		}
		cudaThreadSynchronize();//synchronizing before calculating the full PDF
		Nstart = cfg->Nhist;
		for (unsigned int iType = 0; iType < cfg->Ntype; iType++) {//calculating full PDF by summing partial PDFs
			for (unsigned int jType = iType; jType < cfg->Ntype; jType++, Nstart += cfg->Nhist){
				float multIJ = float(SL[iType] * SL[jType]) / Faverage2;
				for (unsigned int iter = 0; iter < GridSizeMax; iter += GridSize) calcPDFkernel << <GridSize, BlockSize >> > (dPDF + iter*BlockSize, Nstart, cfg->Nhist - iter*BlockSize, multIJ);
				cudaThreadSynchronize();//synchronizing before adding next partial PDF to the full PDF
			}
		}
		hPDF = new float[NPDFh];
		cudaMemcpy(hPDF, dPDF, NPDFh*sizeof(float), cudaMemcpyDeviceToHost);//copying the PDF from the device to the host
		for (unsigned int i = 0; i < NPDFh; i++) (*PDF)[i] = double(hPDF[i]);//converting into double
		delete[] hPDF;
		if (dPDF != NULL) cudaFree(dPDF);
		cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
	    cudaEventElapsedTime(&time, start, stop);
	    cout << "PDF calculation time: " << time/1000 << " s" << endl;
	}
	if ((cfg->scenario == 2) || (cfg->scenario == 4)) {
		cudaEventRecord(start, 0);
		calcInt1DHistCuda(DeviceNUM, I, rij_hist, NatomType, cfg, dFF, SL, dq, Ntot);//calculating the scattering intensity using the pair-distribution histogram
		cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
	    cudaEventElapsedTime(&time, start, stop);
	    cout << "1D pattern calculation time: " << time/1000 << " s" << endl;
	}
	if (rij_hist != NULL) cudaFree(rij_hist);//deallocating memory for pair distribution histogram
}
template <unsigned int BlockSize2D> __global__ void calcIntDebyeKernelXray(float *I, const float *FFi, const float *FFj, const float *q, unsigned int Nq, const float4 *ri, const float4 *rj, unsigned int iMax, unsigned int jMax, bool diag){
	//this kernel calculates the Xray scattering intensity using the Debye formula
	if ((diag) && (blockIdx.x < blockIdx.y)) return; //we need to calculate inter-atomic distances only for j > i, so if we are in the diagonal grid, all the subdiagonal blocks (for which j < i for all threads) do nothing and return
	unsigned int jt = threadIdx.x, it = threadIdx.y;
	unsigned int j = blockIdx.x * BlockSize2D + jt;
	unsigned int iCopy = blockIdx.y * BlockSize2D + jt; //jt!!! memory transaction are performed by the threads of the same warp to coalesce them
	unsigned int i = blockIdx.y * BlockSize2D + it;
	__shared__ float3 ris[BlockSize2D], rjs[BlockSize2D]; //cache arrays for the atomic coordinates (we use float3 here to avoid bank conflicts)
	__shared__ float rij[BlockSize2D][BlockSize2D]; //cache array for inter-atomic distances
	rij[it][jt] = 0; //zeroing inter-atomic distances array
	if ((it == 0) && (j < jMax)) { //copying the atomic coordinates for j-th (column) atoms (only the threads of the first warp (half-warp for CC < 2.0) are used)
		float4 rt = rj[j]; //we cannot copy float4 to float3 directly (without breaking the transaction coalescing) so the temporary variable in local memory is used
		rjs[jt] = make_float3(rt.x, rt.y, rt.z); //and now converting to float3
	}
	if ((it == 2) && (iCopy < iMax)) { //the same for i-th (row) atoms (only the threads of the third warp (first half-warp of the second warp for CC < 2.0) are used)
		float4 rt = ri[iCopy];
		ris[jt] = make_float3(rt.x, rt.y, rt.z);
	}
	__syncthreads(); //synchronizing threads to ensure that the copying is complete
	if (!diag){
		if ((j < jMax) && (i < iMax)) rij[it][jt] = length(ris[it] - rjs[jt]);//calculating distances
	}
	else{//we are in diagonal grid
		if ((j < jMax) && (i < iMax) && (j > i)) rij[it][jt] = length(ris[it] - rjs[jt]);//all the subdiagonal blocks already quit, but we have diagonal blocks (blockIdx.x == blockIdx.y), so we should check if j > i
	}
	__syncthreads();//synchronizing threads to ensure that the calculation of the distances is complete
	iMax = MIN(BlockSize2D, iMax - blockIdx.y * BlockSize2D); //last i-th (row) atom index for the current block
	jMax = MIN(BlockSize2D, jMax - blockIdx.x * BlockSize2D); //last j-th (column) atom index for the current block
	for (unsigned int iterq = 0; iterq < Nq; iterq += SQR(BlockSize2D)) {//if Nq > SQR(BlockSize2D) there will be threads that compute more than one element of the intensity array
		unsigned int iq = iterq + it*BlockSize2D + jt;
		if (iq < Nq) {//checking for array margin
			float lI = 0, qrij;
			float lq = q[iq];//copying the scattering vector modulus to the local memory
			if ((diag) && (blockIdx.x == blockIdx.y)) {//diagonal blocks, j starts from i + 1
				for (i = 0; i < iMax; i++) {
#pragma unroll 8//unrolling to speed up the performance
					for (j = i + 1; j < jMax; j++) {
						qrij = lq*rij[i][j];
						lI += __sinf(qrij) / (qrij + 0.000001f); //scattering intensity without form-factors
					}
				}
			}
			else {//j starts from 0
				for (i = 0; i < iMax; i++) {
#pragma unroll 8
					for (j = 0; j < jMax; j++) {
						qrij = lq*rij[i][j];
						lI += __sinf(qrij) / (qrij + 0.000001f);
					}
				}
			}
			I[Nq * (gridDim.x * blockIdx.y + blockIdx.x) + iq] += 2.f * lI * FFi[iq] * FFj[iq]; //multiplying the intensity by form-factors and storing the results in the global memory (2.f is for j < i part)
		}
	}
}
template <unsigned int BlockSize2D> __global__ void calcIntDebyeKernelNeutron(float *I, float SLIJ, const float *q, unsigned int Nq, const float4 *ri, const float4 *rj, unsigned int iMax, unsigned int jMax, bool diag){
	//this kernel calculates the neutron scattering intensity using the Debye formula
	//see comments in the calcIntDebyeKernelXray() kernel
	if ((diag) && (blockIdx.x < blockIdx.y)) return;
	unsigned int jt = threadIdx.x, it = threadIdx.y;
	unsigned int j = blockIdx.x * BlockSize2D + jt;
	unsigned int iCopy = blockIdx.y * BlockSize2D + jt; //jt!!!
	unsigned int i = blockIdx.y * BlockSize2D + it;
	__shared__ float3 ris[BlockSize2D], rjs[BlockSize2D];
	__shared__ float rij[BlockSize2D][BlockSize2D];
	rij[it][jt] = 0;
	if ((it == 0) && (j < jMax)) {
		float4 rt = rj[j];
		rjs[jt] = make_float3(rt.x, rt.y, rt.z);
	}
	if ((it == 2) && (iCopy < iMax)) {
		float4 rt = ri[iCopy];
		ris[jt] = make_float3(rt.x, rt.y, rt.z);
	}
	__syncthreads();
	if (!diag){
		if ((j < jMax) && (i < iMax)) rij[it][jt] = length(ris[it] - rjs[jt]);
	}
	else{
		if ((j < jMax) && (i < iMax) && (j > i)) rij[it][jt] = length(ris[it] - rjs[jt]);
	}
	__syncthreads();
	iMax = MIN(BlockSize2D, iMax - blockIdx.y * BlockSize2D);
	jMax = MIN(BlockSize2D, jMax - blockIdx.x * BlockSize2D);
	for (unsigned int iterq = 0; iterq < Nq; iterq += SQR(BlockSize2D)) {
		unsigned int iq = iterq + it*BlockSize2D + jt;
		if (iq < Nq) {
			float lI = 0, qrij;
			float lq = q[iq];
			if ((diag) && (blockIdx.x == blockIdx.y)) {
				for (i = 0; i < iMax; i++) {
#pragma unroll 8
					for (j = i + 1; j < jMax; j++) {
						qrij = lq*rij[i][j];
						lI += __sinf(qrij) / (qrij + 0.000001f);
					}
				}
			}
			else {
				for (i = 0; i < iMax; i++) {
#pragma unroll 8
					for (j = 0; j < jMax; j++) {
						qrij = lq*rij[i][j];
						lI += __sinf(qrij) / (qrij + 0.000001f);
					}
				}
			}
			I[Nq * (gridDim.x * blockIdx.y + blockIdx.x) + iq] += 2.f * lI * SLIJ;
		}
	}
}
void calcIntDebyeCuda(unsigned int DeviceNUM, double **I, const calc *cfg, const unsigned int *NatomType, const float4 *ra, const float * const * dFF, vector<double> SL, const float *dq){
	//calculating the Xray and neutron scattering intensity using the Debye formula
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	unsigned int BlockSize2D = BlockSize2Dsmall;//setting block size to 32x32 (default)
	float *dI = NULL, *hI = NULL; //host and device arrays for scattering intensity
	unsigned int Ntot = 0;
	for (unsigned int iType = 0; iType < cfg->Ntype; iType++) Ntot += NatomType[iType]; //calculating total number of atoms
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, DeviceNUM);//getting the device properties
	size_t free, total;
	cuMemGetInfo(&free, &total);//checking the amount of free GPU memory	
	unsigned int GridSizeExecMax = MIN(128, (unsigned int)(sqrtf(0.5f * free / (cfg->q.N*sizeof(float)))));//we use two-dimensional grid here, so checking the amount of free memory is really important 
	unsigned int BlockSize = SQR(BlockSize2D);//total number of threads per block
	unsigned int GFLOPS = GetGFLOPS(deviceProp); //theoretical peak GPU performance
	if (deviceProp.kernelExecTimeoutEnabled)	{//killswitch is enabled, so the time limit should not be exceeded
		const double tmax = 0.2; //maximum kernel time execution in seconds
		const double k = 5.e-8; // t = k * GridSizeExecMax^2 * BlockSize2D^2 * cfg->q.N / GFLOPS
		GridSizeExecMax = MIN((unsigned int)(sqrt(tmax * GFLOPS / (k * cfg->q.N)) / BlockSize2D), GridSizeExecMax);
	}
	unsigned int Isize = SQR(GridSizeExecMax)*cfg->q.N;//total size of the intensity array
	cudaError err=cudaMalloc(&dI, Isize*sizeof(float));//allocating memory for the intensity array and checking for errors
	if (err != cudaSuccess) cout << "Error in calcIntDebyeCuda(), cudaMalloc(dI): " << cudaGetErrorString(err) << endl;
	unsigned int GSzero = MIN(65535, Isize / BlockSize + BOOL(Isize % BlockSize));//grid size for zero1DFloatArrayKernel
	for (unsigned int iter = 0; iter < Isize / BlockSize + BOOL(Isize % BlockSize); iter += GSzero) zero1DFloatArrayKernel << <GSzero, BlockSize >> >(dI + iter*BlockSize, Isize - iter*BlockSize);//zeroing the intensity array
	cudaThreadSynchronize();//synchronizing before calculation starts
	dim3 blockgrid(BlockSize2D, BlockSize2D);
	unsigned int iAtomST = 0, jAtom0, GSadd = cfg->q.N / BlockSize1Dsmall + BOOL(cfg->q.N % BlockSize1Dsmall);//grid size for addIKernelXray/addIKernelNeutron
	bool diag = false;
	for (unsigned int iType = 0; iType < cfg->Ntype; iAtomST += NatomType[iType], iType++) {
		if (cfg->source) addIKernelXray << <GSadd, BlockSize1Dsmall >> > (dI, dFF[iType], cfg->q.N, NatomType[iType]);//adding contribution from diagonal (i==j) elements in Debye sum
		else addIKernelNeutron << <GSadd, BlockSize1Dsmall >> > (dI, cfg->q.N, float(SQR(SL[iType]) * NatomType[iType]));
		cudaThreadSynchronize();//synchronizing before main calculation starts
		unsigned int jAtomST = iAtomST;
		for (unsigned int jType = iType; jType < cfg->Ntype; jAtomST += NatomType[jType], jType++) {
			for (unsigned int iAtom = 0; iAtom < NatomType[iType]; iAtom += BlockSize2D*GridSizeExecMax){
				unsigned int GridSizeExecY = MIN((NatomType[iType] - iAtom) / BlockSize2D + BOOL((NatomType[iType] - iAtom) % BlockSize2D), GridSizeExecMax);//Y-size of grid on current step
				unsigned int iMax = MIN(BlockSize2D*GridSizeExecY, NatomType[iType] - iAtom);//last i-th (row) atom in current grid
				(iType == jType) ? jAtom0 = iAtom : jAtom0 = 0;
				for (unsigned int jAtom = jAtom0; jAtom < NatomType[jType]; jAtom += BlockSize2D*GridSizeExecMax){
					unsigned int GridSizeExecX = MIN((NatomType[jType] - jAtom) / BlockSize2D + BOOL((NatomType[jType] - jAtom) % BlockSize2D), GridSizeExecMax);//X-size of grid on current step
					unsigned int jMax = MIN(BlockSize2D*GridSizeExecX, NatomType[jType] - jAtom);//last j-th (column) atom in current grid
					dim3 grid(GridSizeExecX, GridSizeExecY);
					(iAtomST + iAtom == jAtomST + jAtom) ? diag = true : diag = false;//checking if we are in diagonal grid
					/*float time;
					cudaEvent_t start, stop;
					cudaEventCreate(&start);
					cudaEventCreate(&stop);
					cudaEventRecord(start, 0);*/
					if (cfg->source) {//Xray
						calcIntDebyeKernelXray <BlockSize2Dsmall> << <grid, blockgrid >> > (dI, dFF[iType], dFF[jType], dq, cfg->q.N, ra + iAtomST + iAtom, ra + jAtomST + jAtom, iMax, jMax, diag);
					}
					else {//neutron
						calcIntDebyeKernelNeutron <BlockSize2Dsmall> << <grid, blockgrid >> > (dI, float(SL[iType] * SL[jType]), dq, cfg->q.N, ra + iAtomST + iAtom, ra + jAtomST + jAtom, iMax, jMax, diag);
					}
					cudaThreadSynchronize();//synchronizing before launching next kernel (it will write the data to the same array)
					/*cudaEventRecord(stop, 0);
					cudaEventSynchronize(stop);
					cudaEventElapsedTime(&time, start, stop);
					cout << "calcIntDebyeKernel execution time is: " << time << " ms\n" << endl;*/
				}
			}
		}
	}
	sumIKernel << <GSadd, BlockSize1Dsmall >> >(dI, cfg->q.N, SQR(GridSizeExecMax));//summing intensity copies
	cudaThreadSynchronize();//synchronizing before multiplying intensity by a polarization factor
	if (cfg->PolarFactor) PolarFactor1DKernel << <GSadd, BlockSize1Dsmall >> >(dI, cfg->q.N, dq, float(cfg->lambda));
	hI = new float [cfg->q.N];
	*I = new double[cfg->q.N];
	cudaMemcpy(hI, dI, cfg->q.N*sizeof(float), cudaMemcpyDeviceToHost);//copying the resulting scattering intensity from the device to the host
	for (unsigned int iq = 0; iq<cfg->q.N; iq++) (*I)[iq] = double(hI[iq]) / Ntot;//normalizing
	cudaFree(dI);//deallocating device memory for intensity array
	delete[] hI;
	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cout << "1D pattern calculation time: " << time/1000 << " s" << endl;
}
__global__ void sumIpartialKernel(float *I, float *Ipart, unsigned int Nq, unsigned int Nsum){
	unsigned int iq = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int ipart = blockIdx.y * Nsum * Nq;
	if (iq<Nq) {
		for (unsigned int j = 1; j < Nsum; j++)	I[ipart + iq] += I[ipart + j*Nq + iq];
		Ipart[(blockIdx.y + 1) * Nq + iq] = I[ipart + iq];
	}
}
__global__ void integrateIpartialKernel(float *I, unsigned int Nq, unsigned int Nparts){
	unsigned int iq = blockDim.x * blockIdx.x + threadIdx.x;
	if (iq<Nq) {
		I[iq] = 0;
		for (unsigned int ipart = 1; ipart < Nparts + 1; ipart++)	I[iq] += I[ipart * Nq + iq];
	}
}
void calcIntPartialDebyeCuda(unsigned int DeviceNUM, double **I, const calc *cfg, const unsigned int *NatomType, const float4 *ra, const float * const * dFF, vector<double> SL, const float *dq, const block *Block){
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	unsigned int GridSizeExecMax, BlockSize2D = BlockSize2Dsmall, BlockSize, Nparts = (cfg->Nblocks * (cfg->Nblocks + 1)) / 2;
	float *dI = NULL, *dIpart = NULL, *hI = NULL;
	unsigned int Ntot = 0, *NatomTypeBlock;
	for (unsigned int iType = 0; iType < cfg->Ntype; iType++) Ntot += NatomType[iType];
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, DeviceNUM);
	BlockSize = SQR(BlockSize2D);
	unsigned int GFLOPS = GetGFLOPS(deviceProp); //theoretical peak GPU performance
	size_t free, total;
	cuMemGetInfo(&free, &total);
	GridSizeExecMax = MIN(128, (unsigned int)(sqrtf(0.5f * free / (Nparts * cfg->q.N*sizeof(float)))));
	if (deviceProp.kernelExecTimeoutEnabled)	{
		//killswitch enabled, so the time limit should not be exceeded
		const double tmax = 0.2; //maximum kernel time execution in seconds
		const double k = 5.e-8; // t = k * GridSizeExecMax^2 * BlockSize2D^2 / GFLOPS
		GridSizeExecMax = MIN((unsigned int)(sqrt(tmax * GFLOPS / (k * cfg->q.N)) / BlockSize2D), GridSizeExecMax);
	}
	unsigned int IsizeBlock = SQR(GridSizeExecMax)*cfg->q.N, Isize = Nparts * IsizeBlock;//each block writes to it's own copy of scattering intensity
	cudaError err = cudaMalloc(&dI, Isize*sizeof(float));
	if (err != cudaSuccess) cout << "Error in calcIntPartialDebyeCuda(), cudaMalloc(dI): " << cudaGetErrorString(err) << endl;
	unsigned int GSzero = MIN(65535, Isize / BlockSize + BOOL(Isize % BlockSize));
	for (unsigned int iter = 0; iter < Isize / BlockSize + BOOL(Isize % BlockSize); iter += GSzero) zero1DFloatArrayKernel << <GSzero, BlockSize >> >(dI+iter*BlockSize, Isize - iter*BlockSize);
	cudaThreadSynchronize();
	dim3 blockgrid(BlockSize2D, BlockSize2D);
	unsigned int iAtomST = 0, GSadd = cfg->q.N / BlockSize1Dsmall + BOOL(cfg->q.N % BlockSize1Dsmall), Istart;
	bool diag = false;
	NatomTypeBlock = new unsigned int[cfg->Ntype*cfg->Nblocks];
	for (unsigned int iType = 0; iType < cfg->Ntype; iType++) {
		for (unsigned int iB = 0; iB < cfg->Nblocks; iB++){
			NatomTypeBlock[iType*cfg->Nblocks + iB] = 0;
			for (unsigned int iBtype = 0; iBtype < Block[iB].Nid; iBtype++) {
				if (Block[iB].id[iBtype] == iType) {
					NatomTypeBlock[iType*cfg->Nblocks + iB] = Block[iB].NatomTypeAll[iBtype];
					break;
				}
			}
		}
	}
	for (unsigned int iType = 0; iType < cfg->Ntype; iAtomST += NatomType[iType], iType++) {
		for (unsigned int iB = 0; iB < cfg->Nblocks; iB++){
			Istart = IsizeBlock * (cfg->Nblocks * iB - (iB * (iB + 1)) / 2 +iB);
			if (cfg->source) addIKernelXray << <GSadd, BlockSize1Dsmall >> > (dI + Istart, dFF[iType], cfg->q.N, NatomTypeBlock[iType*cfg->Nblocks + iB]);
			else addIKernelNeutron << <GSadd, BlockSize1Dsmall >> > (dI + Istart, cfg->q.N, float(SQR(SL[iType]) * NatomTypeBlock[iType*cfg->Nblocks + iB]));
		}
		cudaThreadSynchronize();
		unsigned int jAtomST = iAtomST;
		for (unsigned int jType = iType; jType < cfg->Ntype; jAtomST += NatomType[jType], jType++) {
			unsigned int iAtomSB = 0;
			for (unsigned int iB = 0; iB < cfg->Nblocks; iAtomSB += NatomTypeBlock[iType*cfg->Nblocks + iB], iB++) {
				for (unsigned int iAtom = 0; iAtom < NatomTypeBlock[iType*cfg->Nblocks + iB]; iAtom += BlockSize2D*GridSizeExecMax){
					unsigned int GridSizeExecY = MIN((NatomTypeBlock[iType*cfg->Nblocks + iB] - iAtom) / BlockSize2D + BOOL((NatomTypeBlock[iType*cfg->Nblocks + iB] - iAtom) % BlockSize2D), GridSizeExecMax);
					unsigned int iMax = MIN(BlockSize2D*GridSizeExecY, NatomType[iType] - iAtom);
					unsigned int i0 = iAtomST + iAtomSB + iAtom;
					unsigned int jAtomSB = 0;
					for (unsigned int jB = 0; jB < cfg->Nblocks; jAtomSB += NatomTypeBlock[jType*cfg->Nblocks + jB], jB++) {
						(jB>iB) ? Istart = IsizeBlock * (cfg->Nblocks * iB - (iB * (iB + 1)) / 2 + jB) : Istart = IsizeBlock * (cfg->Nblocks * jB - (jB * (jB + 1)) / 2 + iB);
						for (unsigned int jAtom = 0; jAtom < NatomTypeBlock[jType*cfg->Nblocks + jB]; jAtom += BlockSize2D*GridSizeExecMax){
							unsigned int j0 = jAtomST + jAtomSB + jAtom;
							if (j0 >= i0) {
								unsigned int GridSizeExecX = MIN((NatomTypeBlock[jType*cfg->Nblocks + jB] - jAtom) / BlockSize2D + BOOL((NatomTypeBlock[jType*cfg->Nblocks + jB] - jAtom) % BlockSize2D), GridSizeExecMax);
								unsigned int jMax = MIN(BlockSize2D*GridSizeExecX, NatomTypeBlock[jType*cfg->Nblocks + jB] - jAtom);
								dim3 grid(GridSizeExecX, GridSizeExecY);
								(i0 == j0) ? diag = true : diag = false;
								if (cfg->source) {									
									calcIntDebyeKernelXray <BlockSize2Dsmall> << <grid, blockgrid >> > (dI + Istart, dFF[iType], dFF[jType], dq, cfg->q.N, ra + i0, ra + j0, iMax, jMax, diag);
								}
								else {
									calcIntDebyeKernelNeutron <BlockSize2Dsmall> << <grid, blockgrid >> > (dI + Istart, float(SL[iType] * SL[jType]), dq, cfg->q.N, ra + i0, ra + j0, iMax, jMax, diag);
								}
								cudaThreadSynchronize();
							}
						}
					}					
				}
			}			
		}
	}
	delete[] NatomTypeBlock;
	unsigned int IpartialSize = (Nparts + 1) * cfg->q.N;
	cudaMalloc(&dIpart, IpartialSize*sizeof(float));
	dim3 gridAdd(GSadd, Nparts);
	sumIpartialKernel << <gridAdd, BlockSize1Dsmall >> >(dI, dIpart, cfg->q.N, SQR(GridSizeExecMax));
	cudaThreadSynchronize();
	cudaFree(dI);
	integrateIpartialKernel << <GSadd, BlockSize1Dsmall >> > (dIpart, cfg->q.N, Nparts);
	cudaThreadSynchronize();
	dim3 gridPolar(GSadd, Nparts + 1);
	if (cfg->PolarFactor) PolarFactor1DKernel << <gridPolar, BlockSize1Dsmall >> >(dIpart, cfg->q.N, dq, float(cfg->lambda));
	hI = new float[IpartialSize];
	*I = new double[IpartialSize];
	cudaMemcpy(hI, dIpart, IpartialSize*sizeof(float), cudaMemcpyDeviceToHost);
	for (unsigned int i = 0; i<IpartialSize; i++) (*I)[i] = double(hI[i]) / Ntot;
	cudaFree(dIpart);
	delete[] hI;
	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cout << "1D pattern calculation time: " << time/1000 << " s" << endl;
}
int SetDeviceCuda(unsigned int *DeviceNUM){
	//queries CUDA devices, changes DeviceNUM to proper device number if necessary
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	if (!nDevices) {
		cout << "Error: No CUDA devices found." << endl;
		return -1;
	}
	if (*DeviceNUM > 0){
		if (*DeviceNUM <= (unsigned int)nDevices){
			*DeviceNUM-=1;
			cudaSetDevice(*DeviceNUM);
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, *DeviceNUM);
			cout << "Selected CUDA device:" << endl;
			GetGFLOPS(deviceProp, true);
			return 0;
		}
		cout << "Error: Unable to set CUDA device " << *DeviceNUM << ". The total number of CUDA devices is " << nDevices << ".\n";
		cout << "Will use the fastest CUDA device." << endl;
	}
	cout << "The following CUDA devices are found.\n";
	cudaDeviceProp deviceProp;
	unsigned int GFOLPS=0, MaxGFOLPS=0;
	for (int i = 0; i< nDevices; i++) {
		cudaGetDeviceProperties(&deviceProp,i);
		cout << "Device " << i+1 << ":" << endl;
		GFOLPS=GetGFLOPS(deviceProp, true);
		if (GFOLPS>MaxGFOLPS) {
			MaxGFOLPS=GFOLPS;
			*DeviceNUM = (unsigned int) i;
		}
	}
	cout << "Will use CUDA device " << *DeviceNUM+1 << "." << endl;
	cudaSetDevice(*DeviceNUM);
	return 0;
}
void dataCopyCUDA(const double *q, const calc *cfg, const vector < vect3d <double> > *ra, float4 **dra, float ***dFF, float **dq, vector <double*> FF, unsigned int Ntot){
	//copying the main data to the device memory
	if (cfg->scenario != 3) {//we are calculating no only PDFs but the diffraction patterns too
		float *qfloat;//temporary float array for the scattering vector modulus
		qfloat = new float[cfg->q.N];
		for (unsigned int iq = 0; iq < cfg->q.N; iq++) qfloat[iq] = (float)q[iq];//converting scattering vector modulus from double to float
		cudaMalloc(dq, cfg->q.N*sizeof(float));//allocating memory for the scattering vector modulus array
		cudaMemcpy(*dq, qfloat, cfg->q.N*sizeof(float), cudaMemcpyHostToDevice);//copying scattering vector modulus array from the host to the device
		delete[] qfloat;//deleting temporary array
		if (cfg->source) {//Xray source
			*dFF = new float*[cfg->Ntype];//this array will store pointers to the atomic form-factor arrays stored in the device memory
			for (unsigned int iType = 0; iType < cfg->Ntype; iType++){
				(*dFF)[iType] = NULL;
				float *FFfloat;//temporary float array for the atomic form-factor
				FFfloat = new float[cfg->q.N];
				for (unsigned int iq = 0; iq < cfg->q.N; iq++) FFfloat[iq] = float(FF[iType][iq]);//converting form-factors from double to float
				cudaMalloc(&(*dFF)[iType], cfg->q.N*sizeof(float));//allocating device memory for the atomic form-factors
				cudaMemcpy((*dFF)[iType], FFfloat, cfg->q.N*sizeof(float), cudaMemcpyHostToDevice);//copying form-factors from the host to the device
				delete[] FFfloat;//deleting temporary array
			}
		}
	}
	cudaMalloc(dra,Ntot*sizeof(float4));//allocating device memory for the atomic coordinates array
	float4 *hra;//temporary host array for atomic coordinates
	hra=new float4[Ntot];
	unsigned int iAtom = 0;
	for (unsigned int iType = 0; iType < cfg->Ntype; iType++){
		for (vector<vect3d <double> >::const_iterator ri = ra[iType].begin(); ri != ra[iType].end(); ri++, iAtom++){
			hra[iAtom] = make_float4((float)ri->x, (float)ri->y, (float)ri->z, 0);//converting atomic coordinates from vect3d <double> to float4
		}
	}	
	cudaMemcpy(*dra,hra,Ntot*sizeof(float4),cudaMemcpyHostToDevice);//copying atomic coordinates from the host to the device
	delete[] hra;//deleting temporary array
}
void delDataFromDevice(float4 *ra,float **dFF,float *dq, unsigned int Ntype){
	cudaFree(ra);//deallocating device memory for the atomic coordinates array
	if (dq != NULL) cudaFree(dq);//deallocating memory for the scattering vector modulus array
	if (dFF != NULL) {//Xray source
		for (unsigned int i = 0; i < Ntype; i++) if (dFF[i] != NULL) cudaFree(dFF[i]);//deallocating device memory for the atomic form-factors
		delete[] dFF;//deleting pointer array
	}
	cudaDeviceReset();//NVIDIA Profiler works improperly without this
}
#endif
