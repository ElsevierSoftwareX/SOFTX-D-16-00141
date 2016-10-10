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

#define SQR(x) ((x)*(x))
#define BlockSize2D 16
#define SizeR 128
#define PIf 3.14159265f
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define BOOL(x) ((x) ? 1 : 0)

__kernel void zeroInt2DKernel(__global float *I, unsigned int Nq, unsigned int Nfi){
	//zeroing the 2d intensity array 
	unsigned int iq = get_local_size(1)*get_group_id(1) + get_local_id(1), ifi = get_local_size(0)*get_group_id(0) + get_local_id(0);
	if ((iq < Nq) && (ifi < Nfi))	I[iq*Nfi + ifi] = 0;
}
__kernel void zeroAmp2DKernel(__global float *Ar, __global float *Ai, unsigned int Nq, unsigned int Nfi){
	//zeroint real and imaginary parts of the 2d amplitude array 
	unsigned int iq = get_local_size(1)*get_group_id(1) + get_local_id(1), ifi = get_local_size(0)*get_group_id(0) + get_local_id(0);
	if ((iq < Nq) && (ifi < Nfi)){
		Ar[iq*Nfi + ifi] = 0;
		Ai[iq*Nfi + ifi] = 0;
	}
}
__kernel void Sum2DKernel(__global float *I, const __global float *Ar, const __global float *Ai, unsigned int Nq, unsigned int Nfi){
	//calculating the 2d intensity array from real and imaginary parts of the 2d amplitude array
	unsigned int iq = get_local_size(1)*get_group_id(1) + get_local_id(1), ifi = get_local_size(0)*get_group_id(0) + get_local_id(0);
	if ((iq < Nq) && (ifi < Nfi))	I[iq*Nfi + ifi] += SQR(Ar[iq*Nfi + ifi]) + SQR(Ai[iq*Nfi + ifi]);
}
__kernel void Norm2DKernel(__global float *I, unsigned int Nq, unsigned int Nfi, float norm){
	//multiplying the 2d intensity array by a normalizing factor
	unsigned int iq = get_local_size(1)*get_group_id(1) + get_local_id(1), ifi = get_local_size(0)*get_group_id(0) + get_local_id(0);
	if ((iq < Nq) && (ifi < Nfi))	I[iq*Nfi + ifi] *= norm;
}
__kernel void PolarFactor2DKernel(__global float *I, unsigned int Nq, unsigned int Nfi,const __global float *q, float lambda){
	//computing the polarizing factor and mutiplying the 2d intensity array by it
	unsigned int iq = get_local_size(1)*get_group_id(1) + get_local_id(1), ifi = get_local_size(0)*get_group_id(0) + get_local_id(0);
	unsigned int iqCopy = get_local_size(1)*get_group_id(1) + get_local_id(0);
	__local float factor[BlockSize2D];
	if ((get_local_id(1) == 0) && (iqCopy < Nq)) {
		//polarizing factor is computed only by the first BlockSize2D threads of the first wavefront/warp and stored in the shared memory
		float sintheta = q[iqCopy] * (lambda * 0.25f / PIf);
		float cos2theta = 1.f - 2.f * SQR(sintheta);
		factor[get_local_id(0)] = 0.5f * (1.f + SQR(cos2theta));
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if ((iq < Nq) && (ifi < Nfi)) I[iq*Nfi + ifi] *= factor[get_local_id(1)];
}
__kernel void calcInt2DKernelXray(__global float *Ar, __global float *Ai, const __global float *q, unsigned int Nq, unsigned int Nfi, const __global float4 *CS, float lambda, const __global float4 *ra, unsigned int Nst, unsigned int Nfin, const __global float *FF){
	//this kernel computes real and imaginary parts of the 2D Xray scattering amplitude in polar coordinates (q,fi)
	//to avoid bank conflicts for shared memory operations BlockSize2D should be equal to the size of the warp (or half-warp for the devices with the CC < 2.0)
	//SizeR should be a multiple of BlockSize2D
	unsigned int iq = get_local_size(1)*get_group_id(1) + get_local_id(1), ifi = get_local_size(0)*get_group_id(0) + get_local_id(0); //each thread computes only one element of 2D amplitude matrix
	unsigned int iqCopy = get_local_size(1)*get_group_id(1) + get_local_id(0);//copying of the scattering vector modulus array to the shared memory performed by the threads of the same warp
	__local float lFF[BlockSize2D]; //cache array for the atomic from-factors
	__local float qi[BlockSize2D]; //cache array for the scattering vector modulus
	__local float4 r[SizeR]; //cache array for the atomic coordinates
	unsigned int Niter = Nfin / SizeR + BOOL(Nfin % SizeR);//we don't have enough shared memory to load the array of atomic coordinates as a whole, so we do it with iterations
	float4 qv; //scattering vector
	float lAr = 0, lAi = 0, cosfi = 0, sinfi = 0, sintheta = 0, costheta = 0;
	if ((get_local_id(1) == 0) && (iqCopy < Nq)) lFF[get_local_id(0)] = FF[iqCopy]; //loading Xray form-factors to the shared memory
	if ((get_local_id(1) == 4) && (iqCopy < Nq)) qi[get_local_id(0)] = q[iqCopy]; //loading scattering vector modulus to shared memory
	barrier(CLK_LOCAL_MEM_FENCE); //synchronizing after loading to the shared memory
	if ((iq < Nq) && (ifi < Nfi)){//checking the 2d array margins
		float arg = ifi*2.f*PIf / Nfi;
		sinfi = native_sin(arg);
		cosfi = native_cos(arg);
		sintheta = 0.25f*lambda*qi[get_local_id(1)] / PIf; //q = 4pi/lambda*sin(theta)
		costheta = 1.f - SQR(sintheta); //theta in [0, pi/2];
		qv = (float4)(costheta*cosfi, costheta*sinfi, -sintheta, 0)*qi[get_local_id(1)];//computing the scattering vector
		//instead of pre-multiplying the atomic coordinates by the rotational matrix we are pre-multiplying the scattering vector by the transposed rotational matrix (dot(qv,r) will be the same)
		qv = (float4)(dot(qv, CS[0]),dot(qv, CS[1]),dot(qv, CS[2]),0);
	}
	for (unsigned int iter = 0; iter < Niter; iter++){
		unsigned int NiterFin = MIN(Nfin - iter * SizeR, SizeR); //checking for the margins of the atomic coordinates array
		if (get_local_id(1) < SizeR / BlockSize2D) {
			unsigned int iAtom = get_local_id(1)*BlockSize2D + get_local_id(0);
			if (iAtom < NiterFin) r[iAtom] = ra[Nst + iter * SizeR + iAtom]; //loading the atomic coordinates to the shared memory
		}
		barrier(CLK_LOCAL_MEM_FENCE); //synchronizing after loading to shared memory
		if ((iq < Nq) && (ifi < Nfi)){//checking the 2d array margins
			for (unsigned int iAtom = 0; iAtom < NiterFin; iAtom++){
				float arg = dot(qv, r[iAtom]);
				sinfi = native_sin(arg);
				cosfi = native_cos(arg);
				lAr += cosfi; //real part of the amplitute
				lAi += sinfi; //imaginary part of the amplitute
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE); //synchronizing before the next loading starts
	}
	if ((iq < Nq) && (ifi < Nfi)){//checking the 2d array margins
		Ar[iq*Nfi + ifi] += lFF[get_local_id(1)] * lAr; //multiplying the real part of the amplitude by the form-factor and writing the results to the global memory
		Ai[iq*Nfi + ifi] += lFF[get_local_id(1)] * lAi; //doing the same for the imaginary part of the amplitude
	}
}
__kernel void calcInt2DKernelNeutron(__global float *Ar, __global float *Ai, const __global float *q, unsigned int Nq, unsigned int Nfi, const __global float4 *CS, float lambda, const __global float4 *ra, unsigned int Nst, unsigned int Nfin, float SL){
	//this kernel computes real and imaginary parts of the 2D neutron scattering amplitude in polar coordinates (q,fi)
	//see comments in the calcInt2DKernelXray() kernel
	unsigned int iq = get_local_size(1)*get_group_id(1) + get_local_id(1), ifi = get_local_size(0)*get_group_id(0) + get_local_id(0);
	unsigned int iqCopy = get_local_size(1)*get_group_id(1) + get_local_id(0);
	__local float qi[BlockSize2D];
	__local float4 r[SizeR];
	unsigned int Niter = Nfin / SizeR + BOOL(Nfin % SizeR);
	float4 qv;
	float lAr = 0, lAi = 0, cosfi = 0, sinfi = 0, sintheta = 0, costheta = 0;
	if ((get_local_id(1) == 4) && (iqCopy < Nq)) qi[get_local_id(0)] = q[iqCopy];
	barrier(CLK_LOCAL_MEM_FENCE);
	if ((iq < Nq) && (ifi < Nfi)){
		float arg = ifi*2.f*PIf / Nfi;
		sinfi = native_sin(arg);
		cosfi = native_cos(arg);
		sintheta = 0.25f*lambda*qi[get_local_id(1)] / PIf;
		costheta = 1.f - SQR(sintheta);
		qv = (float4)(costheta*cosfi, costheta*sinfi, -sintheta,0)*qi[get_local_id(1)];
		qv = (float4)(dot(qv, CS[0]), dot(qv, CS[1]), dot(qv, CS[2]),0);
	}
	for (unsigned int iter = 0; iter < Niter; iter++){
		unsigned int NiterFin = MIN(Nfin - iter * SizeR, SizeR);
		if (get_local_id(1) < SizeR / BlockSize2D) {
			unsigned int iAtom = get_local_id(1)*BlockSize2D + get_local_id(0);
			if (iAtom < NiterFin) r[iAtom] = ra[Nst + iter * SizeR + iAtom];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if ((iq < Nq) && (ifi < Nfi)){
			for (unsigned int iAtom = 0; iAtom < NiterFin; iAtom++){
				float arg = dot(qv, r[iAtom]);
				sinfi = native_sin(arg);
				cosfi = native_cos(arg);
				lAr += cosfi;
				lAi += sinfi;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if ((iq < Nq) && (ifi < Nfi)){
		Ar[iq*Nfi + ifi] += SL * lAr;
		Ai[iq*Nfi + ifi] += SL * lAi;
	}
}