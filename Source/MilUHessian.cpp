/*
 * MilUHessian.cpp
 *
 *  Created on: 2017/04/25
 *      Author: uemura
 */

#include <MilUHessian.h>

#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <iostream>

MilUHessian3D::MilUHessian3D(MilUData3D &src)
{
	std::vector<int> shape = src.getShape();
	if (shape.at(0) != 5 || shape.at(1) != 5 || shape.at(2) != 5)
	{
		std::cerr << "Data Shape is not 5x5x5." << std::cout;
		throw std::range_error("MilUHessian Constructor");
	}

	this->calcHessianMat(src);

	setEigenValuesToVec();
	std::sort(this->eigenValues.begin(), this->eigenValues.end());

//	std::cout << "Hessian Mtarix :\n" << this->mat << std::endl;
}

MilUHessian3D::~MilUHessian3D()
{

}

void MilUHessian3D::calcHessianMat(MilUData3D &src)
{
	this->mat(0, 0) = this->secondOrderDiff(src, MIL_U_DIFF_XX);
	this->mat(0, 1) = this->secondOrderDiff(src, MIL_U_DIFF_XY);
	this->mat(0, 2) = this->secondOrderDiff(src, MIL_U_DIFF_XZ);
	this->mat(1, 1) = this->secondOrderDiff(src, MIL_U_DIFF_YY);
	this->mat(1, 2) = this->secondOrderDiff(src, MIL_U_DIFF_YZ);
	this->mat(2, 2) = this->secondOrderDiff(src, MIL_U_DIFF_ZZ);

	this->mat(1, 0) = this->mat(0, 1);
	this->mat(2, 0) = this->mat(0, 2);
	this->mat(2, 1) = this->mat(1, 2);
}

double MilUHessian3D::enhanceMassiveStructure(double gamma)
{
	double enhancedValue = 0;
	double l1 = this->eigenValues.at(2), l2 = this->eigenValues.at(1), l3 = this->eigenValues.at(0);
	if (l1 < 0)
	{
		double l2Pl3 = (l2 / l3);
		double l1Pl2 = (l1 / l2);
		enhancedValue = abs(eigenValues.at(2)) * pow(l2Pl3, gamma) * pow(l1Pl2, gamma);
	}

	return enhancedValue;
}

void MilUHessian3D::setEigenValuesToVec()
{
	Eigen::EigenSolver<Eigen::Matrix3d> eig(this->mat);
	auto vals = eig.eigenvalues();

//	std::cout << "Eigen Values :\n" << vals << std::endl;
	std::vector<double> eigenValueVector{vals(0).real(), vals(1).real(), vals(2).real()};

	this->eigenValues = eigenValueVector;
}

double MilUHessian3D::secondOrderDiff(MilUData3D &data, MIL_U_DIFF_ODER order)
{
	double secondOrderDiff = 0;
	switch(order)
	{
	case MIL_U_DIFF_XX:
		secondOrderDiff = this->secondOrderDiffHelper(data, MIL_U_DIFF_X, MIL_U_DIFF_X);
		break;

	case MIL_U_DIFF_XY:
		secondOrderDiff = this->secondOrderDiffHelper(data, MIL_U_DIFF_X, MIL_U_DIFF_Y);
		break;

	case MIL_U_DIFF_XZ:
		secondOrderDiff = this->secondOrderDiffHelper(data, MIL_U_DIFF_X, MIL_U_DIFF_Z);
		break;

	case MIL_U_DIFF_YY:
		secondOrderDiff = this->secondOrderDiffHelper(data, MIL_U_DIFF_Y, MIL_U_DIFF_Y);
		break;

	case MIL_U_DIFF_YZ:
		secondOrderDiff = this->secondOrderDiffHelper(data, MIL_U_DIFF_Y, MIL_U_DIFF_Z);
		break;

	case MIL_U_DIFF_ZZ:
		secondOrderDiff = this->secondOrderDiffHelper(data, MIL_U_DIFF_Z, MIL_U_DIFF_Z);
		break;

	default:
		break;
	}

	return secondOrderDiff;
}

double MilUHessian3D::secondOrderDiffHelper(MilUData3D &data, MIL_U_DIFF_DIRECT firstDir, MIL_U_DIFF_DIRECT secondDir)
{
	MilUData3D diffVolume(data.getShape());
	for (int i = 1; i < 4; i++)
	{
		for (int j = 1; j < 4; j++)
		{
			for (int k = 1; k < 4; k++)
			{
				MilCoordinates coor = {k, j, i};
				diffVolume.at(k, j, i) = this->sobel(data, firstDir, coor);
			}
		}
	}
	MilCoordinates coor = {2, 2, 2};
	return this->sobel(diffVolume, secondDir, coor);
}

double MilUHessian3D::sobel(MilUData3D &data, MIL_U_DIFF_DIRECT direction, MilCoordinates &coor)
{
	if ((coor.x < 1 || coor.x >= data.getShape()[0] - 1) || (coor.y < 1 || coor.y >= data.getShape()[1] - 1) || (coor.z < 1 || coor.z >= data.getShape()[2] - 1))
	{
		throw std::out_of_range("MilUHessian Sobel");
	}

	double diff = 0;

	switch(direction)
	{
	case MIL_U_DIFF_X:
		// sub
		diff += data.at(coor.x - 1, coor.y    , coor.z - 1) * -1;

		diff += data.at(coor.x - 1, coor.y - 1, coor.z    ) * -1;
		diff += data.at(coor.x - 1, coor.y    , coor.z    ) * -2;
		diff += data.at(coor.x - 1, coor.y + 1, coor.z    ) * -1;

		diff += data.at(coor.x - 1, coor.y - 1, coor.z + 1) * -1;

		// add
		diff += data.at(coor.x + 1, coor.y    , coor.z - 1) *  1;

		diff += data.at(coor.x + 1, coor.y - 1, coor.z    ) *  1;
		diff += data.at(coor.x + 1, coor.y    , coor.z    ) *  2;
		diff += data.at(coor.x + 1, coor.y + 1, coor.z    ) *  1;

		diff += data.at(coor.x + 1, coor.y - 1, coor.z + 1) *  1;
		break;

	case MIL_U_DIFF_Y:
		// sub
		diff += data.at(coor.x    , coor.y - 1, coor.z - 1) * -1;

		diff += data.at(coor.x - 1, coor.y - 1, coor.z    ) * -1;
		diff += data.at(coor.x    , coor.y - 1, coor.z    ) * -2;
		diff += data.at(coor.x + 1, coor.y - 1, coor.z    ) * -1;

		diff += data.at(coor.x    , coor.y - 1, coor.z + 1) * -1;

		// add
		diff += data.at(coor.x    , coor.y + 1, coor.z - 1) *  1;

		diff += data.at(coor.x - 1, coor.y + 1, coor.z    ) *  1;
		diff += data.at(coor.x    , coor.y + 1, coor.z    ) *  2;
		diff += data.at(coor.x + 1, coor.y + 1, coor.z    ) *  1;

		diff += data.at(coor.x    , coor.y + 1, coor.z + 1) *  1;
		break;

	case MIL_U_DIFF_Z:
		// sub
		diff += data.at(coor.x    , coor.y - 1, coor.z + 1) * -1;

		diff += data.at(coor.x - 1, coor.y    , coor.z + 1) * -1;
		diff += data.at(coor.x    , coor.y    , coor.z + 1) * -2;
		diff += data.at(coor.x + 1, coor.y    , coor.z + 1) * -1;

		diff += data.at(coor.x    , coor.y + 1, coor.z + 1) * -1;

		// add
		diff += data.at(coor.x    , coor.y - 1, coor.z - 1) *  1;

		diff += data.at(coor.x - 1, coor.y    , coor.z - 1) *  1;
		diff += data.at(coor.x    , coor.y    , coor.z - 1) *  2;
		diff += data.at(coor.x + 1, coor.y    , coor.z - 1) *  1;

		diff += data.at(coor.x    , coor.y + 1, coor.z - 1) *  1;
		break;
	}

	return diff;
}
