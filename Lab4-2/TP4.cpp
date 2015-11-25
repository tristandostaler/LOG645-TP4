#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <CL/opencl.h>
#include <iostream>
using namespace std;

// Host buffers for demo
// *********************************************************************
cl_float *srcB;        // Host buffers for OpenCL test

// OpenCL Vars
cl_context cxGPUContext;        // OpenCL context
cl_command_queue cqCommandQueue;// OpenCL command que
cl_platform_id cpPlatform;      // OpenCL platform
cl_device_id cdDevice;          // OpenCL device
cl_program cpProgram;           // OpenCL program
cl_kernel ckKernel;             // OpenCL kernel
cl_mem cmDevSrcA;               // OpenCL device source buffer A
cl_mem cmDevWrite;                // OpenCL device destination buffer 
cl_mem cmDevSrcB;               // OpenCL device source buffer B 
cl_mem cmDevDst;                // OpenCL device destination buffer 
size_t szGlobalWorkSize;        // 1D var for Total # of work items
size_t szLocalWorkSize;		    // 1D var for # of work items in the work group	
size_t szParmDataBytes;			// Byte size of context information
size_t szKernelLength;			// Byte size of kernel code
cl_int ciErr1, ciErr2;			// Error code var
char* cPathAndName = NULL;      // var for full paths to data, src, etc.
char* cSourceCL = NULL;         // Buffer to hold source for compilation
const char* cExecutableName = NULL;

typedef struct
{
	int nbLignes_n;
	int nbCol_m;
	int nombreDePasDeTemps_np;
	float td;
	float h;
} Arguments;

int iNumElements = 11444777;	// Length of float arrays to process (odd # for illustration)

cl_float* matInitiale;
cl_float* matFinale;
cl_int* matWritable;


// Forward Declarations
// *********************************************************************
//void VectorAddHost(const float* pfData1, const float* pfData2, float* pfResult, int iNumElements);
void Cleanup(int argc, char **argv, int iExitCode);


//////////////////////////////////////////////////////////////////////////////
//! Loads a Program file and prepends the cPreamble to the code.
//!
//! @return the source string if succeeded, 0 otherwise
//! @param cFilename program filename
//! @param cPreamble code that is prepended to the loaded file, typically a set of 
//! #defines or a header
//! @param szFinalLength returned length of the code string
//////////////////////////////////////////////////////////////////////////////
char* oclLoadProgSource(const char* cFilename, const char* cPreamble, size_t* szFinalLength)
{
	// locals
	FILE* pFileStream = NULL;
	size_t szSourceLength;
	// open the OpenCL source code file
	if (fopen_s(&pFileStream, cFilename, "rb") != 0)
	{
		return NULL;
	}
	size_t szPreambleLength = strlen(cPreamble);
	// get the length of the source code
	fseek(pFileStream, 0, SEEK_END);
	szSourceLength = ftell(pFileStream);
	fseek(pFileStream, 0, SEEK_SET);
	// allocate a buffer for the source code string and read it in
	char* cSourceString = (char *)malloc(szSourceLength + szPreambleLength + 1);
	memcpy(cSourceString, cPreamble, szPreambleLength);
	if (fread((cSourceString)+szPreambleLength, szSourceLength, 1, pFileStream) != 1)
	{
		fclose(pFileStream);
		free(cSourceString);
		return 0;
	}
	// close the file and return the total length of the combined (preamble + source) string
	fclose(pFileStream);
	if (szFinalLength != 0)
	{
		*szFinalLength = szSourceLength + szPreambleLength;
	}
	cSourceString[szSourceLength + szPreambleLength] = '\0';
	return cSourceString;
}

Arguments ParseArgs(int argc, char** argv)
{
	Arguments args = {};
	if (argc < 5){
		printf("5 arguments sont nécessaires (en ordre): \n");
		printf("\tn = Le nombre de lignes\n");
		printf("\tm = Le nombre de colonnes\n");
		printf("\tnp = Le nombre de pas de temps\n");
		printf("\ttd = Le temps discrétisé\n");
		printf("\th = La taille d'un côté d'une subdivision\n");
		return args;
	}


	args.nbLignes_n = atoi(argv[1]);
	args.nbCol_m = atoi(argv[2]);
	args.nombreDePasDeTemps_np = atoi(argv[3]);
	args.td = atof(argv[4]);
	args.h = atof(argv[5]);
	return args;
}

void InitMatrices(int n, int m, float* matInitiale, float* matFinale, int* matWritable)
{
	int counter = 0;
	for (size_t i = 0; i< n; i++)
	{
		for (size_t j = 0; j < m; j++)
		{
			matInitiale[counter] = j*(m - j - 1) * i*(n - i - 1);
			matFinale[counter] = j*(m - j - 1) * i*(n - i - 1);
			if (i == 0 || j == 0 || i == n - 1 || j == m - 1)
				matWritable[counter] = 0;
			else
				matWritable[counter] = 1;
			counter++;
		}
	}
}

int main(int argc, char** argv)
{
	/*size_t* szFinalLength = new size_t;
	cPathAndName = "TP4.cl";
	char* source = oclLoadProgSource(cPathAndName, 0, szFinalLength);*/

	Arguments args = ParseArgs(argc, argv);
	// set and log Global and Local work size dimensions
	szLocalWorkSize = args.nbCol_m > args.nbLignes_n ? args.nbCol_m : args.nbLignes_n;
	szGlobalWorkSize = args.nbCol_m * args.nbLignes_n;  // rounded up to the nearest multiple of the LocalWorkSize

	matInitiale = (cl_float*)malloc(szGlobalWorkSize * sizeof(cl_float));
	matFinale = (cl_float*)malloc(szGlobalWorkSize * sizeof(cl_float));
	matWritable = (cl_int*)malloc(szGlobalWorkSize * sizeof(cl_int));
	srcB = (cl_float*)malloc(7 * sizeof(cl_float));

	InitMatrices(args.nbLignes_n, args.nbCol_m, matInitiale, matFinale, matWritable);

	// start logs 
	cExecutableName = "TP4";
	
	// Allocate and initialize host arrays 
	//srcA = (void *)malloc(sizeof(cl_float)* szGlobalWorkSize);
	

	//dst = (void *)malloc(sizeof(cl_float)* szGlobalWorkSize);

	float td_div_h_squ = args.td / (args.h * args.h);
	float one_minus_four_times_tdhh = (1 - 4 * (td_div_h_squ));

	((float*)srcB)[0] = td_div_h_squ;
	((float*)srcB)[1] = one_minus_four_times_tdhh;
	((float*)srcB)[2] = args.nbCol_m;
	((float*)srcB)[3] = args.nbLignes_n;

	//Get an OpenCL platform
	ciErr1 = clGetPlatformIDs(1, &cpPlatform, NULL);

	if (ciErr1 != CL_SUCCESS)
	{
		Cleanup(argc, argv, EXIT_FAILURE);
	}

	//Get the devices
	ciErr1 = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL);
	if (ciErr1 != CL_SUCCESS)
	{
		Cleanup(argc, argv, EXIT_FAILURE);
	}

	//Create the context
	cxGPUContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &ciErr1);
	if (ciErr1 != CL_SUCCESS)
	{
		Cleanup(argc, argv, EXIT_FAILURE);
	}

	// Create a command-queue
	cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevice, 0, &ciErr1);
	if (ciErr1 != CL_SUCCESS)
	{
		Cleanup(argc, argv, EXIT_FAILURE);
	}

	// Allocate the OpenCL buffer memory objects for source and result on the device GMEM
	cmDevSrcA = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_float)* szGlobalWorkSize, NULL, &ciErr1);
	cmDevWrite = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_int)* szGlobalWorkSize, NULL, &ciErr1);
	cmDevSrcB = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_float)* 7, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	cmDevDst = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, sizeof(cl_float)* szGlobalWorkSize, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	if (ciErr1 != CL_SUCCESS)
	{
		Cleanup(argc, argv, EXIT_FAILURE);
	}

	// Read the OpenCL kernel in from source file
	cPathAndName = (char*)malloc(6 * sizeof(char));
	cPathAndName = "TP4.cl";
	cSourceCL = oclLoadProgSource(cPathAndName, "", &szKernelLength);

	// Create the program
	cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&cSourceCL, &szKernelLength, &ciErr1);
	if (ciErr1 != CL_SUCCESS)
	{
		Cleanup(argc, argv, EXIT_FAILURE);
	}

	// Build the program with 'mad' Optimization option
	char* flags = "-cl-fast-relaxed-math";
	ciErr1 = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);
	if (ciErr1 != CL_SUCCESS)
	{
		Cleanup(argc, argv, EXIT_FAILURE);
	}

	// Create the kernel
	ckKernel = clCreateKernel(cpProgram, "HeatTransfer", &ciErr1);
	if (ciErr1 != CL_SUCCESS)
	{
		Cleanup(argc, argv, EXIT_FAILURE);
	}

	// Set the Argument values
	ciErr1 = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void*)&cmDevSrcA);
	ciErr1 |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void*)&cmDevWrite);
	ciErr1 |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void*)&cmDevSrcB);
	ciErr1 |= clSetKernelArg(ckKernel, 3, sizeof(cl_mem), (void*)&cmDevDst);
	if (ciErr1 != CL_SUCCESS)
	{
		Cleanup(argc, argv, EXIT_FAILURE);
	}

	// --------------------------------------------------------
	// Start Core sequence... copy input data to GPU, compute, copy results back

	// Asynchronous write of data to GPU device
	

	ciErr1 = clEnqueueWriteBuffer(cqCommandQueue, cmDevSrcA, CL_FALSE, 0, sizeof(cl_float)* szGlobalWorkSize, matInitiale, 0, NULL, NULL);
	ciErr1 = clEnqueueWriteBuffer(cqCommandQueue, cmDevWrite, CL_FALSE, 0, sizeof(cl_int)* szGlobalWorkSize, matWritable, 0, NULL, NULL);
	if (ciErr1 != CL_SUCCESS)
	{
		Cleanup(argc, argv, EXIT_FAILURE);
	}
	for (int i = 0; i < args.nombreDePasDeTemps_np; ++i)
	{	
		((float*)srcB)[4] = i;
		ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, cmDevSrcB, CL_FALSE, 0, sizeof(cl_float)* 7, srcB, 0, NULL, NULL);
		// Launch kernel
		ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 1, NULL, &szGlobalWorkSize, &szLocalWorkSize, 0, NULL, NULL);
		if (ciErr1 != CL_SUCCESS)
		{
			Cleanup(argc, argv, EXIT_FAILURE);
		}
	}
	
	// Synchronous/blocking read of results, and check accumulated errors
	if (args.nombreDePasDeTemps_np % 2 == 0)
		ciErr1 = clEnqueueReadBuffer(cqCommandQueue, cmDevDst, CL_TRUE, 0, sizeof(cl_float)* szGlobalWorkSize, matFinale, 0, NULL, NULL);
	else
		ciErr1 = clEnqueueReadBuffer(cqCommandQueue, cmDevSrcA, CL_TRUE, 0, sizeof(cl_float)* szGlobalWorkSize, matFinale, 0, NULL, NULL);
	if (ciErr1 != CL_SUCCESS)
	{
		Cleanup(argc, argv, EXIT_FAILURE);
	}
	//--------------------------------------------------------

	// Cleanup and leave
	Cleanup(argc, argv, EXIT_SUCCESS); //(bMatch == shrTRUE) ? EXIT_SUCCESS : EXIT_FAILURE);

	/*for (int i = 0; i < szGlobalWorkSize; ++i)
	{
		std::cout << ((const cl_float*)matFinale)[i] << std::endl;
		if (i % 1000 == 999)
		{
			int a;
			std::cin >> a;
			cin.clear();
			fflush(stdin);
		}

	}*/

	int a;
	std::cin >> a;

	return 0;
}


void Cleanup(int argc, char **argv, int iExitCode)
{
	// Cleanup allocated objects
	/*if (cPathAndName)
		free(cPathAndName);
	if (cSourceCL)
		free(cSourceCL);
	if (ckKernel)
		clReleaseKernel(ckKernel);
	if (cpProgram)
		clReleaseProgram(cpProgram);
	if (cqCommandQueue)
		clReleaseCommandQueue(cqCommandQueue);
	if (cxGPUContext)
		clReleaseContext(cxGPUContext);
	if (cmDevSrcA)
		clReleaseMemObject(cmDevSrcA);
	if (cmDevSrcB)
		clReleaseMemObject(cmDevSrcB);
	if (cmDevDst)
		clReleaseMemObject(cmDevDst);

	// Free host memory
	free(srcA);
	free(srcB);
	free(dst);*/
}

/*
void ExecuteSequentiel(Arguments args, float matInitiale[args.nbLignes_n][args.nbCol_m], float matFinale[args.nbLignes_n][args.nbCol_m], float td_div_h_squ, float one_minus_four_times_tdhh)
{
for (int i = 0; i < args.nombreDePasDeTemps_np; i++)
{
for (int i = 1; i < args.nbLignes_n - 1; i++)
{
for (int j = 1; j < args.nbCol_m - 1; j++)
{
usleep(TEMPS_ATTENTE);
matFinale[i][j] = one_minus_four_times_tdhh
* matInitiale[i][j]
+ td_div_h_squ
* (matInitiale[i-1][j] + matInitiale[i+1][j] + matInitiale[i][j-1] + matInitiale[i][j+1]);
}
}

for (int i = 0; i < args.nbLignes_n; i++)
{
for (int j = 0; j < args.nbCol_m; j++)
{
matInitiale[i][j] = matFinale[i][j];
}
}
}
}
*/