/*
 * MilUHessian.h
 *
 *  Created on: 2017/04/25
 *      Author: uemura
 */

#ifndef MILUHESSIAN3D_H_
#define MILUHESSIAN3D_H_

#include <Eigen/Core>
#include <vector>
#include <MilUMacro.h>
#include <MilUNumpyData.h>
#include <MilUData3D.h>

typedef enum
{
	MIL_U_DIFF_XX,
	MIL_U_DIFF_XY,
	MIL_U_DIFF_XZ,
	MIL_U_DIFF_YY,
	MIL_U_DIFF_YZ,
	MIL_U_DIFF_ZZ
} MIL_U_DIFF_ODER;

typedef enum
{
	MIL_U_DIFF_X,
	MIL_U_DIFF_Y,
	MIL_U_DIFF_Z
} MIL_U_DIFF_DIRECT;

typedef struct
{
	int x;
	int y;
	int z;
} MilCoordinates;

class MilUHessian3D {
public:
	explicit MilUHessian3D(MilUData3D &src);
	virtual ~MilUHessian3D();

	GetMacro(Eigen::MatrixXd, mat, Matrix);
	GetMacro(std::vector<double>, eigenValues, EigenValuesToVec);

	void calcHessianMat(MilUData3D &src);
	double enhanceMassiveStructure(double gamma = 1.0);

private:
	Eigen::Matrix3d mat;
	std::vector<double> eigenValues;

	void setEigenValuesToVec();

	double secondOrderDiff(MilUData3D &data, MIL_U_DIFF_ODER oder);
	double secondOrderDiffHelper(MilUData3D &data, MIL_U_DIFF_DIRECT firstDir, MIL_U_DIFF_DIRECT secondDir);
	double sobel(MilUData3D &data, MIL_U_DIFF_DIRECT direction, MilCoordinates &coor);
};

#endif /* MILUHESSIAN3D_H_ */
