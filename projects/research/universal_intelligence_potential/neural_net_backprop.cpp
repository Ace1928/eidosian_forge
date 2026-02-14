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

// Function to load data from a file into a vector
void load_data(const std::string& filename, std::vector<float>& vec) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    }
    float value;
    while (file >> value) {
        vec.push_back(value);
    }
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
    const float learning_rate = 0.01f;
    const int epochs = 1000;

    // Allocate host memory
    std::vector<float> input;
    std::vector<float> weights1;
    std::vector<float> biases1(hidden_size1);
    std::vector<float> hidden1(hidden_size1);
    std::vector<float> weights2;
    std::vector<float> biases2(hidden_size2);
    std::vector<float> hidden2(hidden_size2);
    std::vector<float> weights3;
    std::vector<float> biases3(output_size);
    std::vector<float> output(output_size);
    std::vector<float> target(output_size);

    
    // Initialize input with specific values for testing
    input = {1.0f, 0.5f, -1.0f};
    target = {0.0f, 1.0f}; // Example target output for testing
    initialize_random(weights1);
    initialize_random(biases1);
    initialize_random(weights2);
    initialize_random(biases2);
    initialize_random(weights3);
    initialize_random(biases3);

    // Gradients
    std::vector<float> weights1_grad(input_size * hidden_size1, 0.0f);
    std::vector<float> biases1_grad(hidden_size1, 0.0f);
    std::vector<float> weights2_grad(hidden_size1 * hidden_size2, 0.0f);
    std::vector<float> biases2_grad(hidden_size2, 0.0f);
    std::vector<float> weights3_grad(hidden_size2 * output_size, 0.0f);
    std::vector<float> biases3_grad(output_size, 0.0f);

    // OpenCL setup
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel_forward, kernel_backward, kernel_update_weights, kernel_update_biases;

    err = clGetPlatformIDs(1, &platform, NULL);
    check_error(err, "clGetPlatformIDs");

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    check_error(err, "clGetDeviceIDs");

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    check_error(err, "clCreateContext");

    queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
    check_error(err, "clCreateCommandQueueWithProperties");

    std::string kernel_code = read_kernel_file("neural_net_backprop.cl");
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

    kernel_forward = clCreateKernel(program, "forward_pass", &err);
    check_error(err, "clCreateKernel forward_pass");
    kernel_backward = clCreateKernel(program, "backward_pass", &err);
    check_error(err, "clCreateKernel backward_pass");
    kernel_update_weights = clCreateKernel(program, "update_weights", &err);
    check_error(err, "clCreateKernel update_weights");
    kernel_update_biases = clCreateKernel(program, "update_biases", &err);
    check_error(err, "clCreateKernel update_biases");

    // Create buffers
    cl_mem bufInput = clCreateBuffer(context, CL_MEM_READ_ONLY, input_size * sizeof(float), NULL, &err);
    check_error(err, "clCreateBuffer bufInput");
    cl_mem bufWeights1 = clCreateBuffer(context, CL_MEM_READ_WRITE, input_size * hidden_size1 * sizeof(float), NULL, &err);
    check_error(err, "clCreateBuffer bufWeights1");
    cl_mem bufBiases1 = clCreateBuffer(context, CL_MEM_READ_WRITE, hidden_size1 * sizeof(float), NULL, &err);
    check_error(err, "clCreateBuffer bufBiases1");
    cl_mem bufHidden1 = clCreateBuffer(context, CL_MEM_READ_WRITE, hidden_size1 * sizeof(float), NULL, &err);
    check_error(err, "clCreateBuffer bufHidden1");
    cl_mem bufWeights2 = clCreateBuffer(context, CL_MEM_READ_WRITE, hidden_size1 * hidden_size2 * sizeof(float), NULL, &err);
    check_error(err, "clCreateBuffer bufWeights2");
    cl_mem bufBiases2 = clCreateBuffer(context, CL_MEM_READ_WRITE, hidden_size2 * sizeof(float), NULL, &err);
    check_error(err, "clCreateBuffer bufBiases2");
    cl_mem bufHidden2 = clCreateBuffer(context, CL_MEM_READ_WRITE, hidden_size2 * sizeof(float), NULL, &err);
    check_error(err, "clCreateBuffer bufHidden2");
    cl_mem bufWeights3 = clCreateBuffer(context, CL_MEM_READ_WRITE, hidden_size2 * output_size * sizeof(float), NULL, &err);
    check_error(err, "clCreateBuffer bufWeights3");
    cl_mem bufBiases3 = clCreateBuffer(context, CL_MEM_READ_WRITE, output_size * sizeof(float), NULL, &err);
    check_error(err, "clCreateBuffer bufBiases3");
    cl_mem bufOutput = clCreateBuffer(context, CL_MEM_READ_WRITE, output_size * sizeof(float), NULL, &err);
    check_error(err, "clCreateBuffer bufOutput");
    cl_mem bufTarget = clCreateBuffer(context, CL_MEM_READ_ONLY, output_size * sizeof(float), NULL, &err);
    check_error(err, "clCreateBuffer bufTarget");
    cl_mem bufWeights1Grad = clCreateBuffer(context, CL_MEM_READ_WRITE, input_size * hidden_size1 * sizeof(float), NULL, &err);
    check_error(err, "clCreateBuffer bufWeights1Grad");
    cl_mem bufBiases1Grad = clCreateBuffer(context, CL_MEM_READ_WRITE, hidden_size1 * sizeof(float), NULL, &err);
    check_error(err, "clCreateBuffer bufBiases1Grad");
    cl_mem bufWeights2Grad = clCreateBuffer(context, CL_MEM_READ_WRITE, hidden_size1 * hidden_size2 * sizeof(float), NULL, &err);
    check_error(err, "clCreateBuffer bufWeights2Grad");
    cl_mem bufBiases2Grad = clCreateBuffer(context, CL_MEM_READ_WRITE, hidden_size2 * sizeof(float), NULL, &err);
    check_error(err, "clCreateBuffer bufBiases2Grad");
    cl_mem bufWeights3Grad = clCreateBuffer(context, CL_MEM_READ_WRITE, hidden_size2 * output_size * sizeof(float), NULL, &err);
    check_error(err, "clCreateBuffer bufWeights3Grad");
    cl_mem bufBiases3Grad = clCreateBuffer(context, CL_MEM_READ_WRITE, output_size * sizeof(float), NULL, &err);
    check_error(err, "clCreateBuffer bufBiases3Grad");

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
    err = clEnqueueWriteBuffer(queue, bufTarget, CL_TRUE, 0, output_size * sizeof(float), target.data(), 0, NULL, NULL);
    check_error(err, "clEnqueueWriteBuffer bufTarget");

    // Set kernel arguments for forward pass
    err = clSetKernelArg(kernel_forward, 0, sizeof(cl_mem), &bufInput);
    check_error(err, "clSetKernelArg 0");
    err = clSetKernelArg(kernel_forward, 1, sizeof(cl_mem), &bufWeights1);
    check_error(err, "clSetKernelArg 1");
    err = clSetKernelArg(kernel_forward, 2, sizeof(cl_mem), &bufBiases1);
    check_error(err, "clSetKernelArg 2");
    err = clSetKernelArg(kernel_forward, 3, sizeof(cl_mem), &bufHidden1);
    check_error(err, "clSetKernelArg 3");
    err = clSetKernelArg(kernel_forward, 4, sizeof(cl_mem), &bufWeights2);
    check_error(err, "clSetKernelArg 4");
    err = clSetKernelArg(kernel_forward, 5, sizeof(cl_mem), &bufBiases2);
    check_error(err, "clSetKernelArg 5");
    err = clSetKernelArg(kernel_forward, 6, sizeof(cl_mem), &bufHidden2);
    check_error(err, "clSetKernelArg 6");
    err = clSetKernelArg(kernel_forward, 7, sizeof(cl_mem), &bufWeights3);
    check_error(err, "clSetKernelArg 7");
    err = clSetKernelArg(kernel_forward, 8, sizeof(cl_mem), &bufBiases3);
    check_error(err, "clSetKernelArg 8");
    err = clSetKernelArg(kernel_forward, 9, sizeof(cl_mem), &bufOutput);
    check_error(err, "clSetKernelArg 9");
    err = clSetKernelArg(kernel_forward, 10, sizeof(int), &input_size);
    check_error(err, "clSetKernelArg 10");
    err = clSetKernelArg(kernel_forward, 11, sizeof(int), &hidden_size1);
    check_error(err, "clSetKernelArg 11");
    err = clSetKernelArg(kernel_forward, 12, sizeof(int), &hidden_size2);
    check_error(err, "clSetKernelArg 12");
    err = clSetKernelArg(kernel_forward, 13, sizeof(int), &output_size);
    check_error(err, "clSetKernelArg 13");

    // Set kernel arguments for backward pass
    err = clSetKernelArg(kernel_backward, 0, sizeof(cl_mem), &bufInput);
    check_error(err, "clSetKernelArg backward 0");
    err = clSetKernelArg(kernel_backward, 1, sizeof(cl_mem), &bufWeights1);
    check_error(err, "clSetKernelArg backward 1");
    err = clSetKernelArg(kernel_backward, 2, sizeof(cl_mem), &bufBiases1);
    check_error(err, "clSetKernelArg backward 2");
    err = clSetKernelArg(kernel_backward, 3, sizeof(cl_mem), &bufHidden1);
    check_error(err, "clSetKernelArg backward 3");
    err = clSetKernelArg(kernel_backward, 4, sizeof(cl_mem), &bufWeights2);
    check_error(err, "clSetKernelArg backward 4");
    err = clSetKernelArg(kernel_backward, 5, sizeof(cl_mem), &bufBiases2);
    check_error(err, "clSetKernelArg backward 5");
    err = clSetKernelArg(kernel_backward, 6, sizeof(cl_mem), &bufHidden2);
    check_error(err, "clSetKernelArg backward 6");
    err = clSetKernelArg(kernel_backward, 7, sizeof(cl_mem), &bufWeights3);
    check_error(err, "clSetKernelArg backward 7");
    err = clSetKernelArg(kernel_backward, 8, sizeof(cl_mem), &bufBiases3);
    check_error(err, "clSetKernelArg backward 8");
    err = clSetKernelArg(kernel_backward, 9, sizeof(cl_mem), &bufOutput);
    check_error(err, "clSetKernelArg backward 9");
    err = clSetKernelArg(kernel_backward, 10, sizeof(cl_mem), &bufTarget);
    check_error(err, "clSetKernelArg backward 10");
    err = clSetKernelArg(kernel_backward, 11, sizeof(cl_mem), &bufWeights1Grad);
    check_error(err, "clSetKernelArg backward 11");
    err = clSetKernelArg(kernel_backward, 12, sizeof(cl_mem), &bufBiases1Grad);
    check_error(err, "clSetKernelArg backward 12");
    err = clSetKernelArg(kernel_backward, 13, sizeof(cl_mem), &bufWeights2Grad);
    check_error(err, "clSetKernelArg backward 13");
    err = clSetKernelArg(kernel_backward, 14, sizeof(cl_mem), &bufBiases2Grad);
    check_error(err, "clSetKernelArg backward 14");
    err = clSetKernelArg(kernel_backward, 15, sizeof(cl_mem), &bufWeights3Grad);
    check_error(err, "clSetKernelArg backward 15");
    err = clSetKernelArg(kernel_backward, 16, sizeof(cl_mem), &bufBiases3Grad);
    check_error(err, "clSetKernelArg backward 16");
    err = clSetKernelArg(kernel_backward, 17, sizeof(int), &input_size);
    check_error(err, "clSetKernelArg backward 17");
    err = clSetKernelArg(kernel_backward, 18, sizeof(int), &hidden_size1);
    check_error(err, "clSetKernelArg backward 18");
    err = clSetKernelArg(kernel_backward, 19, sizeof(int), &hidden_size2);
    check_error(err, "clSetKernelArg backward 19");
    err = clSetKernelArg(kernel_backward, 20, sizeof(int), &output_size);
    check_error(err, "clSetKernelArg backward 20");
    err = clSetKernelArg(kernel_backward, 21, sizeof(float), &learning_rate);
    check_error(err, "clSetKernelArg backward 21");

    // Set kernel arguments for weight updates
    err = clSetKernelArg(kernel_update_weights, 3, sizeof(float), &learning_rate);
    check_error(err, "clSetKernelArg update_weights");

    // Set kernel arguments for bias updates
    err = clSetKernelArg(kernel_update_biases, 3, sizeof(float), &learning_rate);
    check_error(err, "clSetKernelArg update_biases");

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Execute the forward pass
        size_t global_work_size[1] = { std::max({hidden_size1, hidden_size2, output_size}) };
        err = clEnqueueNDRangeKernel(queue, kernel_forward, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
        check_error(err, "clEnqueueNDRangeKernel forward");

        // Execute the backward pass
        err = clEnqueueNDRangeKernel(queue, kernel_backward, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
        check_error(err, "clEnqueueNDRangeKernel backward");

        // Update weights and biases
        size_t weight_sizes[3] = {input_size * hidden_size1, hidden_size1 * hidden_size2, hidden_size2 * output_size};
        cl_mem weight_bufs[3] = {bufWeights1, bufWeights2, bufWeights3};
        cl_mem weight_grad_bufs[3] = {bufWeights1Grad, bufWeights2Grad, bufWeights3Grad};
        for (int i = 0; i < 3; ++i) {
            err = clSetKernelArg(kernel_update_weights, 0, sizeof(cl_mem), &weight_bufs[i]);
            check_error(err, "clSetKernelArg update_weights 0");
            err = clSetKernelArg(kernel_update_weights, 1, sizeof(cl_mem), &weight_grad_bufs[i]);
            check_error(err, "clSetKernelArg update_weights 1");
            err = clSetKernelArg(kernel_update_weights, 2, sizeof(int), &weight_sizes[i]);
            check_error(err, "clSetKernelArg update_weights 2");
            err = clEnqueueNDRangeKernel(queue, kernel_update_weights, 1, NULL, &weight_sizes[i], NULL, 0, NULL, NULL);
            check_error(err, "clEnqueueNDRangeKernel update_weights");
        }

        size_t bias_sizes[3] = {hidden_size1, hidden_size2, output_size};
        cl_mem bias_bufs[3] = {bufBiases1, bufBiases2, bufBiases3};
        cl_mem bias_grad_bufs[3] = {bufBiases1Grad, bufBiases2Grad, bufBiases3Grad};
        for (int i = 0; i < 3; ++i) {
            err = clSetKernelArg(kernel_update_biases, 0, sizeof(cl_mem), &bias_bufs[i]);
            check_error(err, "clSetKernelArg update_biases 0");
            err = clSetKernelArg(kernel_update_biases, 1, sizeof(cl_mem), &bias_grad_bufs[i]);
            check_error(err, "clSetKernelArg update_biases 1");
            err = clSetKernelArg(kernel_update_biases, 2, sizeof(int), &bias_sizes[i]);
            check_error(err, "clSetKernelArg update_biases 2");
            err = clEnqueueNDRangeKernel(queue, kernel_update_biases, 1, NULL, &bias_sizes[i], NULL, 0, NULL, NULL);
            check_error(err, "clEnqueueNDRangeKernel update_biases");
        }

        // Optionally read and display the output and loss every 100 epochs
        if (epoch % 100 == 0) {
            err = clEnqueueReadBuffer(queue, bufOutput, CL_TRUE, 0, output_size * sizeof(float), output.data(), 0, NULL, NULL);
            check_error(err, "clEnqueueReadBuffer bufOutput");

            float loss = 0.0f;
            for (int i = 0; i < output_size; ++i) {
                loss += 0.5f * (output[i] - target[i]) * (output[i] - target[i]); // Mean squared error
            }
            std::cout << "Epoch " << epoch << " Loss: " << loss << std::endl;
        }
    }

    // Read the final results back to host
    err = clEnqueueReadBuffer(queue, bufOutput, CL_TRUE, 0, output_size * sizeof(float), output.data(), 0, NULL, NULL);
    check_error(err, "clEnqueueReadBuffer bufOutput");

    // Display the final output
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
    clReleaseMemObject(bufTarget);
    clReleaseMemObject(bufWeights1Grad);
    clReleaseMemObject(bufBiases1Grad);
    clReleaseMemObject(bufWeights2Grad);
    clReleaseMemObject(bufBiases2Grad);
    clReleaseMemObject(bufWeights3Grad);
    clReleaseMemObject(bufBiases3Grad);
    clReleaseKernel(kernel_forward);
    clReleaseKernel(kernel_backward);
    clReleaseKernel(kernel_update_weights);
    clReleaseKernel(kernel_update_biases);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}

