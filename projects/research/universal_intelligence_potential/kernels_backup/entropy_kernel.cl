// Function to compute entropy for a given probability
inline float compute_entropy(float prob, float alpha_H, float alpha_Pi, float alpha_log) {
    return alpha_H * (-prob * log(alpha_log * prob) * alpha_Pi);
}

// Kernel function to calculate entropy
__kernel void calculate_entropy(__global const float *probabilities, __global const float *alpha_params, __global float *results, const int size) {
    // Define local memory to reduce global memory access latency
    __local float local_alpha_params[3];
    
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);

    // Load alpha parameters into local memory
    if (local_id < 3) {
        local_alpha_params[local_id] = alpha_params[local_id];
    }

    // Ensure all local memory loads are complete
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute entropy for each probability
    if (global_id < size) {
        float prob = probabilities[global_id];
        results[global_id] = compute_entropy(prob, local_alpha_params[0], local_alpha_params[1], local_alpha_params[2]);
    }
}
