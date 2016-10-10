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
#define BlockSize 256
#define PIf 3.14159265f
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define BOOL(x) ((x) ? 1 : 0)
#ifndef CustomInt64atomics
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#else
void atomInc64 (__global uint *counter)
{
	uint old, carry;

	old = atomic_inc(&counter[0]);
	carry = old == 0xFFFFFFFF;
	atomic_add(&counter[1], carry);
}
#endif
__kernel void PolarFactor1DKernel(__global float *I, unsigned int Nq, const __global float *q, float lambda){
	//computes polarizing factor and mutiplies 1d intensity array by it
	unsigned int iq = get_local_size(0)*get_group_id(0) + get_local_id(0);
	if (iq < Nq)	{
		float sintheta = q[iq] * (lambda * 0.25f / PIf);
		float cos2theta = 1.f - 2.f * SQR(sintheta);
		float factor = 0.5f * (1.f + SQR(cos2theta));
		I[iq] *= factor;
	}
}
__kernel void zero1DFloatArrayKernel(__global float *A, unsigned int N){
	//zeroing any 1d float array
	unsigned int i = get_local_size(0)*get_group_id(0) + get_local_id(0);
	if (i<N) A[i] = 0;
}
__kernel void addIKernelXray(__global float *I, unsigned int Nq, const __global float *FFi, unsigned int N) {
	//adding the diagonal elements to the Xray scattering intensity (j==i in the Debye double sum) 
	unsigned int iq = get_local_size(0)*get_group_id(0) + get_local_id(0);
	if (iq < Nq)	{
		float lFF = FFi[iq];
		I[iq] += SQR(lFF) * N;
	}
}
__kernel void addIKernelNeutron(__global float *I, unsigned int Nq, float Add) {
	//adding the diagonal elements to the neutron scattering intensity (j==i in the Debye double sum) 
	unsigned int iq = get_local_size(0)*get_group_id(0) + get_local_id(0);
	if (iq < Nq)	I[iq] += Add;
}
__kernel void sumIKernel(__global float *I, unsigned int Nq, unsigned int Nsum){
	//reducing the grid array of scattering intensity into final array (first Nq elements)
	unsigned int iq = get_local_size(0)*get_group_id(0) + get_local_id(0);
	if (iq<Nq) {
		for (unsigned int j = 1; j < Nsum; j++)	I[iq] += I[j*Nq + iq];
	}
}
__kernel void zeroHistKernel(__global ulong *rij_hist, unsigned int N){
	//zeroing the pair distribution histogram
	unsigned int i = get_local_size(0)*get_group_id(0) + get_local_id(0);
	if (i<N) rij_hist[i] = 0;
}
__kernel void calcPartialRDFkernel(__global float *dPDF, const __global ulong *rij_hist, unsigned int iSt, unsigned int Nhist, float mult) {
	//calculating the partial radial distribution function (RDF) using the pair distribution histogram
	unsigned int i = get_local_size(0)*get_group_id(0) + get_local_id(0);
	if (i < Nhist) dPDF[Nhist + iSt + i] = rij_hist[iSt + i] * mult;
}
__kernel void calcPartialPDFkernel(__global float *dPDF, const __global ulong *rij_hist, unsigned int iSt, unsigned int Nhist, float mult, float bin) {
	//calculating the partial pair distribution function (PDF) using the pair distribution histogram
	unsigned int i = get_local_size(0)*get_group_id(0) + get_local_id(0);
	if (i < Nhist) {
		float r = (i + 0.5f)*bin;
		dPDF[Nhist + iSt + i] = rij_hist[iSt + i] * (mult / SQR(r));
	}
}
__kernel void calcPartialRPDFkernel(__global float *dPDF, const __global ulong *rij_hist, unsigned int iSt, unsigned int Nhist, float mult, float submult, float bin) {
	//calculating the partial reduced pair-distribution function (rPDF) using the pair distribution histogram
	unsigned int i = get_local_size(0)*get_group_id(0) + get_local_id(0);
	if (i < Nhist) {
		float r = (i + 0.5f)*bin;
		dPDF[Nhist + iSt + i] = rij_hist[iSt + i] * (mult / r) - submult * r;
	}
}
__kernel void calcPDFkernel(__global float *dPDF, unsigned int iSt, unsigned int Nhist, float multIJ) {
	//calculating the full PDF by summing the partial PDFs
	unsigned int i = get_local_size(0)*get_group_id(0) + get_local_id(0);
	if (i < Nhist) 	dPDF[i] += dPDF[iSt + i] * multIJ;
}
__kernel void calcHistKernel(const __global float4 *ra, unsigned int i0, unsigned int j0, unsigned int iMax, unsigned int jMax, __global ulong *rij_hist, unsigned int iSt, float bin, unsigned int diag){
	//this kernel calculates the pair-distribution histogram
	if ((diag) && (get_group_id(0) < get_group_id(1))) return; //we need to calculate inter-atomic distances only for j > i, so if we are in the diagonal grid, all the subdiagonal blocks (for which j < i for all threads) do nothing and return
	unsigned int jt = get_local_id(0), it = get_local_id(1);
	unsigned int j = get_group_id(0) * BlockSize2D + jt;
	unsigned int iCopy = get_group_id(1) * BlockSize2D + jt; //jt!!! memory transaction are performed by the threads of the same warp to coalesce them
	unsigned int i = get_group_id(1) * BlockSize2D + it;
	__local float4 ris[BlockSize2D], rjs[BlockSize2D];
	if ((it == 0) && (j + j0 < jMax)) { //copying atomic coordinates for j-th (column) atoms
		rjs[jt] = ra[j0 + j];
	}
	if ((it == 4) && (iCopy + i0 < iMax)) { //the same for i-th (row) atoms
		ris[jt] = ra[i0 + iCopy];
	}
	barrier(CLK_LOCAL_MEM_FENCE);//sync to ensure that copying is complete
	if (!diag){
		if ((j + j0 < jMax) && (i + i0 < iMax)) {
			float rij = sqrt(SQR(ris[it].x - rjs[jt].x) + SQR(ris[it].y - rjs[jt].y) + SQR(ris[it].z - rjs[jt].z)); //calculate distance
			unsigned int index = (unsigned int)(rij / bin); //get the index of histogram bin
#ifndef CustomInt64atomics
			atom_inc(&rij_hist[iSt + index]); //add +1 to histogram bin
#else
			atomInc64(&rij_hist[iSt + index]);
#endif
		}
	}
	else{//we are in diagonal grid
		if ((j + j0 < jMax) && (i + i0 < iMax) && (j > i)) {//all the subdiagonal blocks already quit, but we have diagonal blocks  (blockIdx.x == blockIdx.y), so we should check if j > i
			float rij = sqrt(SQR(ris[it].x - rjs[jt].x) + SQR(ris[it].y - rjs[jt].y) + SQR(ris[it].z - rjs[jt].z));
			unsigned int index = (unsigned int)(rij / bin);
#ifndef CustomInt64atomics
			atom_inc(&rij_hist[iSt + index]);
#else
			atomInc64(&rij_hist[iSt + index]);
#endif
		}
	}
}
__kernel void calcIntHistKernelXray(__global float *I, const __global float *FFi, const __global float *FFj, const  __global float *q, unsigned int Nq, const __global ulong *rij_hist, unsigned int iSt, unsigned int iBinSt, unsigned int Nhist, unsigned int MaxBinsPerBlock, float bin){
	//this kernel calculates the Xray scattering intensity using the pair-distribution histogram
	unsigned int iBegin = iBinSt + get_group_id(0) * MaxBinsPerBlock;//first index for histogram bin to process
	unsigned int iEnd = MIN(Nhist, iBegin + MaxBinsPerBlock);//last index for histogram bin to process
	for (unsigned int iterq = 0; iterq < (Nq / get_local_size(0)) + BOOL(Nq % get_local_size(0)); iterq++) {//if Nq > blockDim.x there will be threads that compute more than one element of the intensity array
		unsigned int iq = iterq*get_local_size(0) + get_local_id(0);//index of the intensity array element
		if (iq < Nq) {//checking for the array margin
			float lI = 0, qrij;
			float lq = q[iq];//copying the scattering vector modulus to the local memory
			for (unsigned int i = iBegin; i < iEnd; i++) {//looping over the histogram bins
				ulong Nrij = rij_hist[iSt + i];
				if (Nrij){
					qrij = lq * (i + 0.5f)*bin;//distance that corresponds to the current histogram bin
					lI += (Nrij * native_sin(qrij)) / (qrij + 0.000001f);//scattering intensity without form factors
				}
			}
			float lFFij = 2.f * FFi[iq] * FFj[iq];
			I[get_group_id(0)*Nq + iq] += lI * lFFij;//multiplying intensity by form-factors and storing the results in global memory
		}
	}
}
__kernel void calcIntHistKernelNeutron(__global float *I, float SLij, const  __global float *q, unsigned int Nq, const __global ulong *rij_hist, unsigned int iSt, unsigned int iBinSt, unsigned int Nhist, unsigned int MaxBinsPerBlock, float bin){
	//this kernel calculates the Neutron scattering intensity using the pair-distribution histogram
	unsigned int iBegin = iBinSt + get_group_id(0) * MaxBinsPerBlock;
	unsigned int iEnd = MIN(Nhist, iBegin + MaxBinsPerBlock);
	for (unsigned int iterq = 0; iterq < (Nq / get_local_size(0)) + BOOL(Nq % get_local_size(0)); iterq++) {
		unsigned int iq = iterq*get_local_size(0) + get_local_id(0);
		if (iq < Nq) {
			float lI = 0, qrij;
			float lq = q[iq];
			for (unsigned int i = iBegin; i < iEnd; i++) {
				ulong Nrij = rij_hist[iSt + i];
				if (Nrij){
					qrij = lq * (i + 0.5f)*bin;//distance that corresponds to the current histogram bin
					lI += (Nrij * native_sin(qrij)) / (qrij + 0.000001f);//scattering intensity without form factors
				}
			}
			I[get_group_id(0)*Nq + iq] += 2.f * lI * SLij;//multiplying intensity by form-factors and storing the results in global memory
		}
	}
}