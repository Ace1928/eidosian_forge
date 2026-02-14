#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <algorithm>

// Utility function to check for OpenCL errors
void check_error(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        std::cerr << "Error during operation '" << operation << "': " << err << std::endl;
        exit(1);
    }
}

// Function to read kernel file
std::string read_kernel_file(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

// Function to initialize random weights and biases
void initialize_random(std::vector<float>& vec) {
    for (auto& v : vec) {
        v = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() {
    // Define network sizes
    const int input_size = 3;
    const int hidden_size1 = 5;
    const int hidden_size2 = 4;
    const int output_size = 2;

    // Allocate and initialize host memory
    std::vector<float> input(input_size, 0.0f);
    std::vector<float> weights1(input_size * hidden_size1);
    std::vector<float> biases1(hidden_size1);
    std::vector<float> hidden1(hidden_size1);
    std::vector<float> weights2(hidden_size1 * hidden_size2);
    std::vector<float> biases2(hidden_size2);
    std::vector<float> hidden2(hidden_size2);
    std::vector<float> weights3(hidden_size2 * output_size);
    std::vector<float> biases3(output_size);
    std::vector<float> output(output_size);

    // Initialize input with specific values for testing
    input = {1.0f, 0.5f, -1.0f};
    initialize_random(weights1);
    initialize_random(biases1);
    initialize_random(weights2);
    initialize_random(biases2);
    initialize_random(weights3);
    initialize_random(biases3);

    // OpenCL setup
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

    queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
    check_error(err, "clCreateCommandQueueWithProperties");

    std::string kernel_code = read_kernel_file("neural_net.cl");
    const char* kernel_source = kernel_code.c_str();
    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    check_error(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
        std::cerr << "Build log:\n" << log.data() << std::endl;
        check_error(err, "clBuildProgram");
    }

    kernel = clCreateKernel(program, "forward_pass", &err);
    check_error(err, "clCreateKernel");

    // Create buffers
    cl_mem bufInput = clCreateBuffer(context, CL_MEM_READ_ONLY, input_size * sizeof(float), NULL, &err);
    check_error(err, "clCreateBuffer bufInput");
    cl_mem bufWeights1 = clCreateBuffer(context, CL_MEM_READ_ONLY, input_size * hidden_size1 * sizeof(float), NULL, &err);
    check_error(err, "clCreateBuffer bufWeights1");
    cl_mem bufBiases1 = clCreateBuffer(context, CL_MEM_READ_ONLY, hidden_size1 * sizeof(float), NULL, &err);
    check_error(err, "clCreateBuffer bufBiases1");
    cl_mem bufHidden1 = clCreateBuffer(context, CL_MEM_READ_WRITE, hidden_size1 * sizeof(float), NULL, &err);
    check_error(err, "clCreateBuffer bufHidden1");
    cl_mem bufWeights2 = clCreateBuffer(context, CL_MEM_READ_ONLY, hidden_size1 * hidden_size2 * sizeof(float), NULL, &err);
    check_error(err, "clCreateBuffer bufWeights2");
    cl_mem bufBiases2 = clCreateBuffer(context, CL_MEM_READ_ONLY, hidden_size2 * sizeof(float), NULL, &err);
    check_error(err, "clCreateBuffer bufBiases2");
    cl_mem bufHidden2 = clCreateBuffer(context, CL_MEM_READ_WRITE, hidden_size2 * sizeof(float), NULL, &err);
    check_error(err, "clCreateBuffer bufHidden2");
    cl_mem bufWeights3 = clCreateBuffer(context, CL_MEM_READ_ONLY, hidden_size2 * output_size * sizeof(float), NULL, &err);
    check_error(err, "clCreateBuffer bufWeights3");
    cl_mem bufBiases3 = clCreateBuffer(context, CL_MEM_READ_ONLY, output_size * sizeof(float), NULL, &err);
    check_error(err, "clCreateBuffer bufBiases3");
    cl_mem bufOutput = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size * sizeof(float), NULL, &err);
    check_error(err, "clCreateBuffer bufOutput");

    // Write data to buffers
    err = clEnqueueWriteBuffer(queue, bufInput, CL_TRUE, 0, input_size * sizeof(float), input.data(), 0, NULL, NULL);
    check_error(err, "clEnqueueWriteBuffer bufInput");
    err = clEnqueueWriteBuffer(queue, bufWeights1, CL_TRUE, 0, input_size * hidden_size1 * sizeof(float), weights1.data(), 0, NULL, NULL);
    check_error(err, "clEnqueueWriteBuffer bufWeights1");
    err = clEnqueueWriteBuffer(queue, bufBiases1, CL_TRUE, 0, hidden_size1 * sizeof(float), biases1.data(), 0, NULL, NULL);
    check_error(err, "clEnqueueWriteBuffer bufBiases1");
    err = clEnqueueWriteBuffer(queue, bufWeights2, CL_TRUE, 0, hidden_size1 * hidden_size2 * sizeof(float), weights2.data(), 0, NULL, NULL);
    check_error(err, "clEnqueueWriteBuffer bufWeights2");
    err = clEnqueueWriteBuffer(queue, bufBiases2, CL_TRUE, 0, hidden_size2 * sizeof(float), biases2.data(), 0, NULL, NULL);
    check_error(err, "clEnqueueWriteBuffer bufBiases2");
    err = clEnqueueWriteBuffer(queue, bufWeights3, CL_TRUE, 0, hidden_size2 * output_size * sizeof(float), weights3.data(), 0, NULL, NULL);
    check_error(err, "clEnqueueWriteBuffer bufWeights3");
    err = clEnqueueWriteBuffer(queue, bufBiases3, CL_TRUE, 0, output_size * sizeof(float), biases3.data(), 0, NULL, NULL);
    check_error(err, "clEnqueueWriteBuffer bufBiases3");

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufInput);
    check_error(err, "clSetKernelArg 0");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufWeights1);
    check_error(err, "clSetKernelArg 1");
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufBiases1);
    check_error(err, "clSetKernelArg 2");
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufHidden1);
    check_error(err, "clSetKernelArg 3");
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &bufWeights2);
    check_error(err, "clSetKernelArg 4");
    err = clSetKernelArg(kernel, 5, sizeof(cl_mem), &bufBiases2);
    check_error(err, "clSetKernelArg 5");
    err = clSetKernelArg(kernel, 6, sizeof(cl_mem), &bufHidden2);
    check_error(err, "clSetKernelArg 6");
    err = clSetKernelArg(kernel, 7, sizeof(cl_mem), &bufWeights3);
    check_error(err, "clSetKernelArg 7");
    err = clSetKernelArg(kernel, 8, sizeof(cl_mem), &bufBiases3);
    check_error(err, "clSetKernelArg 8");
    err = clSetKernelArg(kernel, 9, sizeof(cl_mem), &bufOutput);
    check_error(err, "clSetKernelArg 9");
    err = clSetKernelArg(kernel, 10, sizeof(int), &input_size);
    check_error(err, "clSetKernelArg 10");
    err = clSetKernelArg(kernel, 11, sizeof(int), &hidden_size1);
    check_error(err, "clSetKernelArg 11");
    err = clSetKernelArg(kernel, 12, sizeof(int), &hidden_size2);
    check_error(err, "clSetKernelArg 12");
    err = clSetKernelArg(kernel, 13, sizeof(int), &output_size);
    check_error(err, "clSetKernelArg 13");

    // Execute the kernel
    size_t global_work_size[1] = { std::max({hidden_size1, hidden_size2, output_size}) };
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
    check_error(err, "clEnqueueNDRangeKernel");

    // Read the results back to host
    err = clEnqueueReadBuffer(queue, bufOutput, CL_TRUE, 0, output_size * sizeof(float), output.data(), 0, NULL, NULL);
    check_error(err, "clEnqueueReadBuffer bufOutput");

    // Display the output
    for (int i = 0; i < output_size; ++i) {
        std::cout << "Output[" << i << "]: " << output[i] << std::endl;
    }

    // Cleanup
    clReleaseMemObject(bufInput);
    clReleaseMemObject(bufWeights1);
    clReleaseMemObject(bufBiases1);
    clReleaseMemObject(bufHidden1);
    clReleaseMemObject(bufWeights2);
    clReleaseMemObject(bufBiases2);
    clReleaseMemObject(bufHidden2);
    clReleaseMemObject(bufWeights3);
    clReleaseMemObject(bufBiases3);
    clReleaseMemObject(bufOutput);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
