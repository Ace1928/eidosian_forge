// Save this as matmul.cpp
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

const char* matmul_kernel_code = R"(
__kernel void matmul(__global float* A, __global float* B, __global float* C, int N) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    float sum = 0.0;
    for (int i = 0; i < N; ++i) {
        sum += A[row * N + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
}
)";

void check_error(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error during operation '%s': %d\n", operation, err);
        exit(1);
    }
}

int main() {
    const int N = 1024;
    size_t data_size = N * N * sizeof(float);
    float* A = (float*)malloc(data_size);
    float* B = (float*)malloc(data_size);
    float* C = (float*)malloc(data_size);

    for (int i = 0; i < N * N; i++) {
        A[i] = (float)(rand() % 1000) / 1000.0f;
        B[i] = (float)(rand() % 1000) / 1000.0f;
    }

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    err = clGetPlatformIDs(1, &platform, NULL);
    check_error(err, "clGetPlatformIDs");

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    check_error(err, "clGetDeviceIDs");

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    check_error(err, "clCreateContext");

    // Use clCreateCommandQueueWithProperties instead of the deprecated clCreateCommandQueue
    cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    queue = clCreateCommandQueueWithProperties(context, device, properties, &err);
    check_error(err, "clCreateCommandQueueWithProperties");

    program = clCreateProgramWithSource(context, 1, &matmul_kernel_code, NULL, &err);
    check_error(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    check_error(err, "clBuildProgram");

    kernel = clCreateKernel(program, "matmul", &err);
    check_error(err, "clCreateKernel");

    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size, NULL, &err);
    check_error(err, "clCreateBuffer bufA");
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size, NULL, &err);
    check_error(err, "clCreateBuffer bufB");
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size, NULL, &err);
    check_error(err, "clCreateBuffer bufC");

    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, data_size, A, 0, NULL, NULL);
    check_error(err, "clEnqueueWriteBuffer bufA");
    err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, data_size, B, 0, NULL, NULL);
    check_error(err, "clEnqueueWriteBuffer bufB");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    check_error(err, "clSetKernelArg 0");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    check_error(err, "clSetKernelArg 1");
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    check_error(err, "clSetKernelArg 2");
    err = clSetKernelArg(kernel, 3, sizeof(int), &N);
    check_error(err, "clSetKernelArg 3");

    size_t global_work_size[2] = { N, N };
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    check_error(err, "clEnqueueNDRangeKernel");

    err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, data_size, C, 0, NULL, NULL);
    check_error(err, "clEnqueueReadBuffer bufC");

    clFinish(queue);

    // Output some values for verification
    for (int i = 0; i < 10; i++) {
        printf("%f ", C[i]);
    }
    printf("\n");

    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(A);
    free(B);
    free(C);

    return 0;
}
