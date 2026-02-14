// Function to compute operational efficiency
inline float compute_operational_efficiency(float P, float E, float alpha_O, float alpha_P, float alpha_E) {
    if (E == 0) {
        return 0.0;  // Avoid division by zero
    } else {
        return alpha_O * (alpha_P * P / (alpha_E * E));
    }
}

// Kernel function to calculate operational efficiency
__kernel void calculate_operational_efficiency(__global const float *P, __global const float *E, __global const float *alpha_params, __global float *result) {
    // Define local memory to reduce global memory access latency
    __local float local_alpha_params[3];
    __local float local_P;
    __local float local_E;

    int global_id = get_global_id(0);
    int local_id = get_local_id(0);

    // Load alpha parameters into local memory
    if (local_id < 3) {
        local_alpha_params[local_id] = alpha_params[local_id];
    }

    // Load P and E into local memory
    if (local_id == 0) {
        local_P = P[0];
        local_E = E[0];
    }

    // Ensure all local memory loads are complete
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute operational efficiency
    if (global_id == 0) {
        float efficiency = compute_operational_efficiency(local_P, local_E, local_alpha_params[0], local_alpha_params[1], local_alpha_params[2]);
        // Store the result using atomic operation to avoid race conditions
        atomic_xchg(result, efficiency);
    }
}
