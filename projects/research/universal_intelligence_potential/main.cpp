#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

// OpenCL kernel code
const char* kernel2Greedy = R"(
__kernel void parallelGreedy2(__global long* denominators, int numDenominators, double numberToExchange, __global char* result) {
    for (int i = 0; i < numDenominators - 1; i++) {
        for (int j = i + 1; j < numDenominators; j++) {
            if (denominators[i] < denominators[j]) {
                long temp = denominators[i];
                denominators[i] = denominators[j];
                denominators[j] = temp;
            }
        }
    }

    result[0] = '\0';

    for (int i = 0; i < numDenominators && numberToExchange > 0; i++) {
        if (numberToExchange >= denominators[i]) {
            int biggestNumberToExchangeInLoop = (int)(numberToExchange / denominators[i]);
            char temp[100];
            snprintf(temp, sizeof(temp), "%ld cash x%d\n", denominators[i], biggestNumberToExchangeInLoop);

            int offset = 0;
            while (result[offset] != '\0') {
                offset++;
            }

            for (int j = 0; temp[j] != '\0'; j++) {
                result[offset++] = temp[j];
            }
            result[offset] = '\0';

            numberToExchange = round(100 * (numberToExchange - (biggestNumberToExchangeInLoop * denominators[i]))) / 100.0;
        }
    }
}
)";

int main() {
    std::cout << "Type a number of instances you want to create:" << std::endl;
    int numberOfInstances2;
    std::cin >> numberOfInstances2;

    std::vector<long> denominators;
    std::cout << "How many numbers you would like to add?" << std::endl;
    int numberToAdd;
    std::cin >> numberToAdd;
    for (int i = 0; i < numberToAdd; i++) {
        std::cout << "Add next number:" << std::endl;
        long nextNumber;
        std::cin >> nextNumber;
        denominators.push_back(nextNumber);
    }

    std::cout << "Enter the number to exchange:" << std::endl;
    double numberToExchange;
    std::cin >> numberToExchange;

    try {
        // Get all platforms (drivers)
        std::vector<cl::Platform> all_platforms;
        cl::Platform::get(&all_platforms);
        if (all_platforms.size() == 0) {
            std::cout << " No platforms found. Check OpenCL installation!\n";
            return 1;
        }
        cl::Platform default_platform = all_platforms[0];
        std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

        // Get default device of the default platform
        std::vector<cl::Device> all_devices;
        default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
        if (all_devices.size() == 0) {
            std::cout << " No devices found. Check OpenCL installation!\n";
            return 1;
        }
        cl::Device default_device = all_devices[0];
        std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

        // Create a context
        cl::Context context(default_device);

        // Create a program
        cl::Program::Sources sources;
        sources.push_back({kernel2Greedy, strlen(kernel2Greedy)});
        cl::Program program(context, sources);

        // Build the program
        try {
            program.build({default_device});
        } catch (const cl::Error& e) {
            std::cerr << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << std::endl;
            throw e;
        }

        // Create buffers
        cl::Buffer bufferDenominators(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(long) * denominators.size(), denominators.data());
        cl::Buffer bufferResult(context, CL_MEM_WRITE_ONLY, sizeof(char) * 1000 * numberOfInstances2);
        cl::Buffer bufferNumberToExchange(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double), &numberToExchange);
        cl::Buffer bufferNumberOfDenominators(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), &numberToAdd);

        // Create the kernel
        cl::Kernel kernel(program, "parallelGreedy2");

        // Set kernel arguments
        kernel.setArg(0, bufferDenominators);
        kernel.setArg(1, bufferNumberOfDenominators);
        kernel.setArg(2, bufferNumberToExchange);
        kernel.setArg(3, bufferResult);

        // Create a command queue
        cl::CommandQueue queue(context, default_device);

        // Run the kernel
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(numberOfInstances2), cl::NullRange);
        queue.finish();

        // Copy the data back
        std::vector<char> result(numberOfInstances2 * 1000);
        queue.enqueueReadBuffer(bufferResult, CL_TRUE, 0, sizeof(char) * 1000 * numberOfInstances2, result.data());

        // Print results
        for (int i = 0; i < numberOfInstances2; ++i) {
            std::cout << "Result for instance " << i << ":" << std::endl;
            std::cout << std::string(result.begin() + (i * 1000), result.begin() + ((i + 1) * 1000)) << std::endl;
        }
    } catch (cl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")" << std::endl;
        return 1;
    }

    return 0;
}