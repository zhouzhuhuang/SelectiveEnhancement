/*
 * MilUHessianGPU.h
 *
 *  Created on: 2017/06/22
 *      Author: uemura
 */

#ifndef MILUHESSIAN3D_GPU_H_
#define MILUHESSIAN3D_GPU_H_

#include <vector>

#include <MilUMacro.h>
#include <MilUCudaData.h>
#include <MilUCuNumpyData.h>

enum MIL_U_DIFF_DIRECT_GPU
{
	MIL_U_GPU_DIFF_X,
	MIL_U_GPU_DIFF_Y,
	MIL_U_GPU_DIFF_Z
};

enum MIL_U_DIFF_ORDER_GPU
{
	MIL_U_GPU_DIFF_XX,
	MIL_U_GPU_DIFF_XY,
	MIL_U_GPU_DIFF_XZ,
	MIL_U_GPU_DIFF_YY,
	MIL_U_GPU_DIFF_YZ,
	MIL_U_GPU_DIFF_ZZ
};

__global__ void enhancement(CudaDataElements<double> src, CudaDataElements<double> dst, double gamma);
__global__ void checkJacobi(CudaDataElements<double> srcMat);
__global__ void sobelOnGPU(CudaDataElements<double> srcVol, MIL_U_DIFF_DIRECT_GPU direction);

__device__ double secondOrderDiff(double *srcVol, MIL_U_DIFF_ORDER_GPU order);
__device__ double secondOrderDiffHelper(double *srcVol, MIL_U_DIFF_DIRECT_GPU first, MIL_U_DIFF_DIRECT_GPU second);
__device__ double sobel(double *mat, MIL_U_DIFF_DIRECT_GPU direction, int3 center, int3 matSize);
__device__ double getMatMax(double mat[][3], int2 *index);
__device__ void jacobi3D(double mat[][3], int maxIter);

#endif /* MILUHESSIAN3D_H_ */
