#include <MilUHessianGPU.h>

__global__ void enhancement(CudaDataElements<double> src, CudaDataElements<double> dst, double gamma, CudaDataElements<bool> mask)
{
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int tid = tx + bx * blockDim.x;

	while(tid < src.cuSize)
	{
		int z = int(floor(tid / (double) (src.cuShape[2] * src.cuShape[1])));
		int residum = (tid) % (src.cuShape[2] * src.cuShape[1]);
		int y = int(floor(residum / (double) src.cuShape[2]));
		int x = int(residum % src.cuShape[2]);

		if(!mask.at(tid))
		{
			tid += blockDim.x * gridDim.x;
			continue;
		}

		if((x >= 2 && x < src.cuShape[2] - 2) && (y >= 2 && y < src.cuShape[1] - 2) && (z >= 2 && z < src.cuShape[0] - 2))
		{
			double tempVol[125];

			for(int i = -2; i < 3; ++i)
			for(int j = -2; j < 3; ++j)
			for(int k = -2; k < 3; ++k)
			{
				int index = (i + z)*src.cuShape[2]*src.cuShape[1] + (j + y)*src.cuShape[2] + (k + x);
				tempVol[5*5*(i + 2) + 5*(j + 2) + (k + 2)] = src.getData(index);
			}

			double diffXX, diffXY, diffXZ, diffYY, diffYZ, diffZZ;
			diffXX = secondOrderDiff(tempVol, MIL_U_GPU_DIFF_XX);
			diffXY = secondOrderDiff(tempVol, MIL_U_GPU_DIFF_XY);
			diffXZ = secondOrderDiff(tempVol, MIL_U_GPU_DIFF_XZ);
			diffYY = secondOrderDiff(tempVol, MIL_U_GPU_DIFF_YY);
			diffYZ = secondOrderDiff(tempVol, MIL_U_GPU_DIFF_YZ);
			diffZZ = secondOrderDiff(tempVol, MIL_U_GPU_DIFF_ZZ);

			double hessianMat[3][3];
			hessianMat[0][0] = diffXX;
			hessianMat[0][1] = diffXY;
			hessianMat[0][2] = diffXZ;

			hessianMat[1][0] = diffXY;
			hessianMat[1][1] = diffYY;
			hessianMat[1][2] = diffYZ;

			hessianMat[2][0] = diffXZ;
			hessianMat[2][1] = diffYZ;
			hessianMat[2][2] = diffZZ;

			jacobi3D(hessianMat, 1000);

			double enhancedVal = 0;
			double lambda[3];
			for(int i = 0; i < 3; ++i)
			{
				lambda[i] = hessianMat[i][i];
			}
			for(int i = 0; i < 2; ++i)
			for(int j = 2; j > i; --j)
			{
				if(lambda[j] > lambda[j - 1])
				{
					double temp = lambda[j];
					lambda[j] = lambda[j - 1];
					lambda[j - 1] = temp;
				}
			}

			if(lambda[0] < 0)
			{
				double l2Pl3 = (lambda[1] / lambda[2]);
				double l1Pl2 = (lambda[0] / lambda[1]);
				enhancedVal = fabs(lambda[2]) * pow(l2Pl3, gamma) * pow(l1Pl2, gamma);
			}
			dst.setData(tid, enhancedVal);
		}

		tid += blockDim.x * gridDim.x;
	}
}

__global__ void checkJacobi(CudaDataElements<double> srcMat)
{
	double mat[3][3];

	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int tid = tx + bx * blockDim.x;

	while(tid < srcMat.cuSize)
	{
		for(int i = 0; i < 3; ++i)
		for(int j = 0; j < 3; ++j)
		{
			mat[i][j] = srcMat.getData(j + i*3);
		}

		jacobi3D(mat, 1000);

		for(int i = 0; i < 3; ++i)
		for(int j = 0; j < 3; ++j)
		{
			 srcMat.setData(j + 3*i, mat[i][j]);
		}

		tid += blockDim.x * gridDim.x;
	}
}

__global__ void sobelOnGPU(CudaDataElements<double> srcVol, MIL_U_DIFF_DIRECT_GPU direction)
{
	double tempVol[27];
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int tid = tx + bx * blockDim.x;

	if(tid == 1)
	{
		for(int i = 0; i < 27; ++i)
		{
			tempVol[i] = srcVol.cuPtr[i];
		}

		int3 center = {1, 1, 1};
		int3 matSize = {3, 3, 3};
		sobel(tempVol, direction, center, matSize);
	}
}

__device__ double sobel(double *mat, MIL_U_DIFF_DIRECT_GPU direction, int3 center, int3 matSize)
{
	double diff = 0;

	double val[3][3][3];
	for(int i = center.z - 1, z = 0; i < center.z + 2; ++i, ++z)
	for(int j = center.y - 1, y = 0; j < center.y + 2; ++j, ++y)
	for(int k = center.x - 1, x = 0; k < center.x + 2; ++k, ++x)
	{
		val[z][y][x] = mat[k + matSize.x * j + matSize.x*matSize.y * i];
	}

	switch(direction)
	{
	case MIL_U_GPU_DIFF_X:
		// sub
		diff += val[0][1][0] * -1;

		diff += val[1][0][0] * -1;
		diff += val[1][1][0] * -2;
		diff += val[1][2][0] * -1;

		diff += val[2][0][0] * -1;

		// add
		diff += val[0][1][2] *  1;

		diff += val[1][0][2] *  1;
		diff += val[1][1][2] *  2;
		diff += val[1][2][2] *  1;

		diff += val[2][0][2] *  1;
		break;

	case MIL_U_GPU_DIFF_Y:
		// sub
		diff += val[0][0][1] * -1;

		diff += val[1][0][0] * -1;
		diff += val[1][0][1] * -2;
		diff += val[1][0][2] * -1;

		diff += val[2][0][1] * -1;

		// add
		diff += val[0][2][1] *  1;

		diff += val[1][2][0] *  1;
		diff += val[1][2][1] *  2;
		diff += val[1][2][2] *  1;

		diff += val[2][2][1] *  1;
		break;

	case MIL_U_GPU_DIFF_Z:
		// sub
		diff += val[2][0][1] * -1;

		diff += val[2][1][0] * -1;
		diff += val[2][1][1] * -2;
		diff += val[2][1][2] * -1;

		diff += val[2][2][1] * -1;

		// add
		diff += val[0][0][1] *  1;

		diff += val[0][1][0] *  1;
		diff += val[0][1][1] *  2;
		diff += val[0][1][2] *  1;

		diff += val[0][2][1] *  1;
		break;
	}

	return diff;
}

__device__ double secondOrderDiff(double *srcVol, MIL_U_DIFF_ORDER_GPU order)
{
	double secondOrderDiff;
	switch(order)
	{
	case MIL_U_GPU_DIFF_XX:
		secondOrderDiff = secondOrderDiffHelper(srcVol, MIL_U_GPU_DIFF_X, MIL_U_GPU_DIFF_X);
		break;

	case MIL_U_GPU_DIFF_XY:
		secondOrderDiff = secondOrderDiffHelper(srcVol, MIL_U_GPU_DIFF_X, MIL_U_GPU_DIFF_Y);
		break;

	case MIL_U_GPU_DIFF_XZ:
		secondOrderDiff = secondOrderDiffHelper(srcVol, MIL_U_GPU_DIFF_X, MIL_U_GPU_DIFF_Z);
		break;

	case MIL_U_GPU_DIFF_YY:
		secondOrderDiff = secondOrderDiffHelper(srcVol, MIL_U_GPU_DIFF_Y, MIL_U_GPU_DIFF_Y);
		break;

	case MIL_U_GPU_DIFF_YZ:
		secondOrderDiff = secondOrderDiffHelper(srcVol, MIL_U_GPU_DIFF_Y, MIL_U_GPU_DIFF_Z);
		break;

	case MIL_U_GPU_DIFF_ZZ:
		secondOrderDiff = secondOrderDiffHelper(srcVol, MIL_U_GPU_DIFF_Z, MIL_U_GPU_DIFF_Z);
		break;

	default:
		break;
	}

	return secondOrderDiff;
}

__device__ double secondOrderDiffHelper(double *srcVol, MIL_U_DIFF_DIRECT_GPU first, MIL_U_DIFF_DIRECT_GPU second)
{
	double tempVol[27];

	for(int i = 1; i < 4; ++i)
	for(int j = 1; j < 4; ++j)
	for(int k = 1; k < 4; ++k)
	{
		int3 center = {k, j, i};
		int3 matSize = {5, 5, 5};
		tempVol[(k - 1) + 3*(j - 1) + 3*3*(i - 1)] = sobel(srcVol, first, center, matSize);
	}

	int3 center = {1, 1, 1};
	int3 matSize = {3, 3, 3};
	return sobel(tempVol, second, center, matSize);
}

__device__ double getMatMax(double mat[][3], int2 *index)
{
	// initialize variables
	double max = 0.;
	index->x = 1;
	index->y = 0;

	for(int i = 0; i < 3; i++)
	for(int j = i + 1; j < 3; j++)
	{
		double curVal = fabs(mat[i][j]);
		if(max < curVal)
		{
			max = curVal;
			index->x = j;
			index->y = i;
		}
	}

	return max;
}

__device__ void jacobi3D(double mat[][3], int maxIter)
{
	float eigenVecs[3][3];
	for(int i = 0; i < 3; i++)
	for(int j = 0; j < 3; j++)
	{
		if(i == j)
		{
			eigenVecs[i][j] = 1.;
		}
		else
		{
			eigenVecs[i][j] = 0.;
		}
	}

	for(int iter = 0; iter < maxIter; ++iter)
	{
		int2 index;
		double max = getMatMax(mat, &index);
		if(max < 0.00001)
		{
			break;
		}

		double app = mat[index.y][index.y];
		double apq = mat[index.y][index.x];
		double aqq = mat[index.x][index.x];

		double alpha = (app - aqq) / 2.;
		double beta = -apq;
		double gamma = fabs(alpha) / sqrt(alpha * alpha + beta * beta);

		double s = sqrt((1 - gamma) / 2.);
		double c = sqrt((1 + gamma) / 2.);
		if(alpha * beta < 0)
		{
			s = -s;
		}

		for(int i = 0; i < 3; ++i)
		{
			double temp = c * mat[index.y][i] - s * mat[index.x][i];
			mat[index.x][i] = s * mat[index.y][i] + c * mat[index.x][i];
			mat[index.y][i] = temp;
		}

		for(int i = 0; i < 3; ++i)
		{
			mat[i][index.y] = mat[index.y][i];
			mat[i][index.x] = mat[index.x][i];
		}

		mat[index.y][index.y] = c*c * app + s*s * aqq - 2*s*c * apq;
		mat[index.y][index.x] = s*c * (app - aqq) + (c*c - s*s) * apq;
		mat[index.x][index.y] = s*c * (app - aqq) + (c*c - s*s) * apq;
		mat[index.x][index.x] = s*s *app + c*c*aqq + 2*s*c*apq;

		for(int i = 0; i < 3; ++i)
		{
			double temp = c * eigenVecs[i][index.y] - s * eigenVecs[i][index.x];
			eigenVecs[i][index.x] = s * eigenVecs[i][index.y] + c * eigenVecs[i][index.x];
			eigenVecs[i][index.y] = temp;
		}
	}
}
