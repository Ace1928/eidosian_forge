// Function to compute time based on temporal scale
inline float compute_time(float temporal_scale, float alpha_t, float alpha_Temporal_Scale) {
    return alpha_t * (alpha_Temporal_Scale * temporal_scale);
}

// Kernel function to calculate time
__kernel void calculate_time(__global const float *temporal_scale, __global const float *alpha_params, __global float *result) {
    // Define local memory to reduce global memory access latency
    __local float local_alpha_params[2];
    __local float local_temporal_scale;

    int global_id = get_global_id(0);
    int local_id = get_local_id(0);

    // Load alpha parameters into local memory
    if (local_id < 2) {
        local_alpha_params[local_id] = alpha_params[local_id];
    }

    // Load temporal scale into local memory
    if (local_id == 0) {
        local_temporal_scale = temporal_scale[0];
    }

    // Ensure all local memory loads are complete
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute time
    if (global_id == 0) {
        float time_value = compute_time(local_temporal_scale, local_alpha_params[0], local_alpha_params[1]);
        // Store the result using atomic operation to avoid race conditions
        atomic_xchg(result, time_value);
    }
}
