// Function to compute volume based on spatial scale
inline float compute_volume(float spatial_scale, float alpha_Volume, float alpha_Spatial_Scale) {
    return alpha_Volume * (alpha_Spatial_Scale * spatial_scale);
}

// Kernel function to calculate volume
__kernel void calculate_volume(__global const float *spatial_scale, __global const float *alpha_params, __global float *result) {
    // Define local memory to reduce global memory access latency
    __local float local_alpha_params[2];
    __local float local_spatial_scale;

    int global_id = get_global_id(0);
    int local_id = get_local_id(0);

    // Load alpha parameters into local memory
    if (local_id < 2) {
        local_alpha_params[local_id] = alpha_params[local_id];
    }

    // Load spatial scale into local memory
    if (local_id == 0) {
        local_spatial_scale = spatial_scale[0];
    }

    // Ensure all local memory loads are complete
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute volume
    if (global_id == 0) {
        float volume_value = compute_volume(local_spatial_scale, local_alpha_params[0], local_alpha_params[1]);
        // Store the result using atomic operation to avoid race conditions
        atomic_xchg(result, volume_value);
    }
}
