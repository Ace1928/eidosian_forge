// Define constants for common operations to avoid magic numbers
#define GROUP_SIZE 256

// Function to compute adaptability
inline float compute_adaptability(float adaptation_rate, float alpha_A, float alpha_Adaptation_Rate) {
    return alpha_A * (alpha_Adaptation_Rate * adaptation_rate);
}

// Kernel function to calculate adaptability
__kernel void calculate_adaptability(__global const float *adaptation_rate, __global const float *alpha_params, __global float *result) {
    // Define local memory to reduce global memory access
    __local float local_alpha_params[2];
    __local float local_adaptation_rate;

    int global_id = get_global_id(0);
    int local_id = get_local_id(0);

    // Load alpha parameters into local memory
    if (local_id < 2) {
        local_alpha_params[local_id] = alpha_params[local_id];
    }

    // Load adaptation rate into local memory
    if (local_id == 0) {
        local_adaptation_rate = adaptation_rate[0];
    }

    // Ensure all local memory loads are complete
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute adaptability using the first work-item in the work-group
    if (local_id == 0) {
        float adaptability = compute_adaptability(local_adaptation_rate, local_alpha_params[0], local_alpha_params[1]);
        // Store the result using atomic operation to avoid race conditions
        atomic_xchg(result, adaptability);
    }
}
