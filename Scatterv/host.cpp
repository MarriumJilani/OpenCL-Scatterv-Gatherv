/*

Sources: http://www.eriksmistad.no/getting-started-with-opencl-and-gpu-computing/

*/

// openCL headers

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif



#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_PROCESSES 10
#define MIN_ELEMENTS 1000
#define MAX_ELEMENTS 10000

#define MAX_SOURCE_SIZE (0x100000)



int main(int argc, char** argv) {

	//data size which will be generated will be between max size and min size
	srand(time(NULL));
	int data_size = rand() % (MAX_ELEMENTS - MIN_ELEMENTS + 1) + MIN_ELEMENTS;
	int sendcounts[NUM_PROCESSES];
	int displacements[NUM_PROCESSES];

	// Allocate memories for input arrays and output array.
	int* data = (int*)malloc(data_size * sizeof(int));
	int* local_sums = (int*)malloc(NUM_PROCESSES * sizeof(int));
	int global_sum = 0;

	//fill with random data
	
	for (int i = 0; i < data_size; i++)
	{
		data[i] = (int)(rand() % 10);
	}

	int s = 0;
	for (int i = 0; i < data_size; i++)
	{
		s+= data[i];
		
	}
	printf("\nActual Sum of random data: %d\n ", s);
	//printf("\Data size: %d\n ", data_size);

	//random sendcounts
	int remaining = data_size;
	for (int i = 0; i < NUM_PROCESSES; i++)
	{
		if (i == NUM_PROCESSES - 1)//if last tou assign all
		{
			sendcounts[i] = remaining;
		}
		else
		{
			sendcounts[i] = rand() % remaining;
			remaining -= sendcounts[i];
		}
	}

	//randomly generate displacements
	int offset = 0;
	for (int i = 0; i < NUM_PROCESSES; i++)
	{
		displacements[i] = offset;
		offset += sendcounts[i];
	}

	int coordinating_process = rand() % NUM_PROCESSES;
	printf("Co ordinating process: %d\n ", coordinating_process);
	// Load kernel from file kernel.cl

	FILE* kernelFile;
	char* kernelSource;
	size_t kernelSize;

	kernelFile = fopen("kernel.cl", "r");

	if (!kernelFile) {

		fprintf(stderr, "No file named kernel.cl was found\n");

		exit(-1);

	}
	kernelSource = (char*)malloc(MAX_SOURCE_SIZE);
	kernelSize = fread(kernelSource, 1, MAX_SOURCE_SIZE, kernelFile);
	fclose(kernelFile);

	// Getting platform and device information
	cl_platform_id platformId = NULL;
	cl_device_id deviceID = NULL;
	cl_uint retNumDevices;
	cl_uint retNumPlatforms;
	cl_int ret = clGetPlatformIDs(1, &platformId, &retNumPlatforms);
	char* value;
	size_t valueSize;

	ret = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &deviceID, &retNumDevices);
	clGetDeviceInfo(deviceID, CL_DEVICE_NAME, 0, NULL, &valueSize);
	value = (char*)malloc(valueSize);
	clGetDeviceInfo(deviceID, CL_DEVICE_NAME, valueSize, value, NULL);
	printf("Device: %s\n", value);
	free(value);

	// Creating context.
	cl_context context = clCreateContext(NULL, 1, &deviceID, NULL, NULL, &ret);


	// Creating command queue
	cl_command_queue commandQueue = clCreateCommandQueue(context, deviceID, 0, &ret);

	// Memory buffers for each array
	cl_mem sendbuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, data_size * sizeof(int), data, &ret);
	cl_mem rcvbuf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, NUM_PROCESSES * sendcounts[0] * sizeof(int), NULL, &ret);

	cl_mem sendcountbuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, NUM_PROCESSES * sizeof(int), sendcounts, &ret);
	cl_mem displacementbuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, NUM_PROCESSES * sizeof(int), displacements, &ret);
	cl_mem localsum_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, NUM_PROCESSES * sizeof(int) , NULL, &ret);
	cl_mem coordinating_process_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) , &coordinating_process, &ret);

	// Copy lists to memory buffers
	ret = clEnqueueWriteBuffer(commandQueue, sendbuf, CL_TRUE, 0, data_size * sizeof(int), data, 0, NULL, NULL);;
	ret = clEnqueueWriteBuffer(commandQueue, sendcountbuf, CL_TRUE, 0, NUM_PROCESSES * sizeof(int), &sendcounts, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(commandQueue, displacementbuf, CL_TRUE, 0, NUM_PROCESSES * sizeof(int), &displacements, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(commandQueue, coordinating_process_buffer, CL_TRUE, 0, sizeof(int),&coordinating_process, 0, NULL, NULL);

	// Create program from kernel source
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, (const size_t*)&kernelSize, &ret);

	// Build program
	ret = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);

	// Create kernel
	cl_kernel kernel = clCreateKernel(program, "Scatterv", &ret);


	// Set arguments for kernel
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&sendbuf);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&rcvbuf);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&sendcountbuf);
	ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&displacementbuf);
	ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&localsum_buf);
	ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&coordinating_process_buffer);



	// Execute the kernel
	size_t globalItemSize = 10;
	size_t localItemSize = 1; // globalItemSize has to be a multiple of localItemSize. 1024/64 = 16 
	ret = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalItemSize, &localItemSize, 0, NULL, NULL);

	// Read from device back to host.

	ret = clEnqueueReadBuffer(commandQueue, localsum_buf, CL_TRUE, 0, NUM_PROCESSES * sizeof(int), local_sums, 0, NULL, NULL);
	//printf("Received:%d\n",  local_sums);
	// Write result
	
	for (int i=0; i<NUM_PROCESSES; ++i) {

		//printf("%d:Local sum:%d\n",i, local_sums[i]);
		global_sum += local_sums[i];
	}
	
	printf("Global sum:%d", global_sum);

	// Clean up, release memory.
	ret = clFlush(commandQueue);
	ret = clFinish(commandQueue);
	ret = clReleaseCommandQueue(commandQueue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(sendbuf);
	ret = clReleaseMemObject(rcvbuf);
	ret = clReleaseMemObject(sendcountbuf);
	ret = clReleaseMemObject(displacementbuf);
	ret = clReleaseMemObject(localsum_buf);

	ret = clReleaseContext(context);
	free(data);
	free(local_sums);
	//free(C);

	return 0;

}