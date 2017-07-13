#include <algorithm>
#include <boost/numpy.hpp>
#include <math.h>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>
#include <iostream>

#include <MilUDataBase.h>
#include <MilUNumpyData.h>
#include <MilUCuNumpyData.h>
#include <MilUHessian.h>
#include <MilUHessianGPU.h>

namespace np = boost::numpy;
namespace bp = boost::python;
namespace spd = spdlog;

auto logger = spdlog::stderr_color_mt("Selective Enhance");

//gauss = lambda sigma, x, y, z : np.exp(-1 * ((x**2 + y**2 + z**2) / (2.0 * (sigma ** 2))))
double gauss(double x, double y, double z, double sigma)
{
	double gaussVal = 0;
	gaussVal = exp(-1 * (x * x + y * y + z * z)/ (2.0 * (sigma * sigma)));
	return gaussVal;
}

double cpuSelectiveEnhancement(np::ndarray inputVolume, double gamma)
{
	if(inputVolume.get_dtype() != np::dtype::get_builtin<double>())
	{
		logger->error("Invalid Argument.");
		throw std::invalid_argument("cpuSEFilter");
	}
	MilUNPdouble inputNumpy(inputVolume);
	MilUData3D inputData(inputNumpy);
	MilUHessian3D hesse(inputData);

	return hesse.enhanceMassiveStructure(gamma);
}

np::ndarray cpuSelectiveEnhancementVoxelwise(np::ndarray inputVolume, double gamma)
{
	if(inputVolume.get_dtype() != np::dtype::get_builtin<double>())
	{
		logger->error("Invalid Argument.");
		throw std::invalid_argument("cpuSEFilter");
	}
	MilUNPdouble inputNumpy(inputVolume);
	MilUData3D inputData(inputNumpy);

	MilUNPdouble outputNumpy(inputNumpy.getShape());
	MilUData3D voi(5, 5, 5);

	for(int i = 2; i < inputData.getShape()[0] - 2; i++)
	for(int j = 2; j < inputData.getShape()[1] - 2; j++)
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

	return outputNumpy.getDataAsNumpy();
}

np::ndarray gpuSelectiveEnhancementVoxelwise(np::ndarray inputVolume, double gamma, np::ndarray mask)
{
	if(inputVolume.get_dtype() != np::dtype::get_builtin<double>() || mask.get_dtype() != np::dtype::get_builtin<bool>())
	{
		logger->error("Invalid Argument.");
		throw std::invalid_argument("gpuSEFilter");
	}
	MilUCuNPdouble inputNpArray(inputVolume);
	MilUCuNPdouble outputNpArray(inputNpArray.getShape());
	MilUCuNPbool maskNP(mask);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int blocks = prop.multiProcessorCount;

	inputNpArray.copyHostToDevice();
	outputNpArray.copyHostToDevice();
	enhancement<<<2*blocks, 512>>>(inputNpArray.getCudaData(), outputNpArray.getCudaData(), gamma, maskNP.getCudaData());
	outputNpArray.copyDeviceToHost();

	return outputNpArray.getDataAsNumpy();
}

BOOST_PYTHON_MODULE(pycuSEFilter)
{
	Py_Initialize();
	np::initialize();
	bp::def("gauss", gauss);
	bp::def("cpuSEFilter", cpuSelectiveEnhancement);
	bp::def("cpuSEFilterVW", cpuSelectiveEnhancementVoxelwise);
	bp::def("gpuSEFilterVW", gpuSelectiveEnhancementVoxelwise);
}
