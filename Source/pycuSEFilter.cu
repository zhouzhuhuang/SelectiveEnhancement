#include <algorithm>
#include <boost/numpy.hpp>
#include <math.h>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>
#include <iostream>

#include <MilUDataBase.h>
#include <MilUNumpyData.h>
#include <MilUHessian.h>

namespace np = boost::numpy;
namespace bp = boost::python;
namespace spd = spdlog;

double cpuSelectiveEnhancement(np::ndarray inputVolume, double gamma)
{
	auto logger = spdlog::stderr_color_mt("Selective Enhance");
	if(inputVolume.get_dtype() != np::dtype::get_builtin<double>())
	{
		logger->error("Invalid Argument.");
		throw std::invalid_argument("cpuSEFilter");
	}
	MilUNumpyData inputNumpy(inputVolume);
	MilUData3D inputData(inputNumpy);
	MilUHessian3D hesse(inputData);

	return hesse.enhanceMassiveStructure(gamma);
}

np::ndarray cpuSelectiveEnhancementVoxelwise(np::ndarray inputVolume, double gamma)
{
	auto logger = spdlog::stderr_color_mt("Selective Enhance VW");
	if(inputVolume.get_dtype() != np::dtype::get_builtin<double>())
	{
		logger->error("Invalid Argument.");
		throw std::invalid_argument("cpuSEFilter");
	}
	MilUNumpyData inputNumpy(inputVolume);
	MilUData3D inputData(inputNumpy);

	MilUNumpyData outputNumpy(inputNumpy.getShape());
	MilUData3D voi(5, 5, 5);

	std::cout << outputNumpy.getShape()[0] << ", " << outputNumpy.getShape()[1] << ", " << outputNumpy.getShape()[2] << " "
			  << inputData.getShape()[0] << ", " << inputData.getShape()[1] << ", " << inputData.getShape()[2];

	for(int i = 2; i < inputData.getShape()[0] - 2; i++){
	for(int j = 2; j < inputData.getShape()[1] - 2; j++){
	for(int k = 2; k < inputData.getShape()[2] - 2; k++)
	{
		for(int z = k - 2, voiZ = 0; z <= k + 2; z++, voiZ++)
		for(int y = j - 2, voiY = 0; y <= j + 2; y++, voiY++)
		for(int x = i - 2, voiX = 0; x <= i + 2; x++, voiX++)
		{
			voi.at(voiX, voiY, voiZ) = inputData.at(x, y, z);
		}

		MilUHessian3D hesse(voi);
		double enhancedVal = hesse.enhanceMassiveStructure(gamma);
		outputNumpy.at(std::vector<int>{i, j, k}) = enhancedVal;
	}
	}
	std::cout << i << std::endl;
	}

	return outputNumpy.getDataAsNumpy();
}


BOOST_PYTHON_MODULE(pycuSEFilter)
{
	Py_Initialize();
	np::initialize();
	bp::def("cpuSEFilter", cpuSelectiveEnhancement);
	bp::def("cpuSEFilterVW", cpuSelectiveEnhancementVoxelwise);
}
