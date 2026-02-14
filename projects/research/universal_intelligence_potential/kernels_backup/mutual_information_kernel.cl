// Function to compute mutual information
inline float compute_mutual_information(float H_X, float H_Y, float H_XY, float alpha_I, float alpha_HX, float alpha_HY, float alpha_HXY) {
    return alpha_I * (alpha_HX * H_X + alpha_HY * H_Y - alpha_HXY * H_XY);
}

// Kernel function to calculate mutual information
__kernel void calculate_mutual_information(__global const float *H_X, __global const float *H_Y, __global const float *H_XY, __global const float *alpha_params, __global float *result) {
    // Define local memory to reduce global memory access latency
    __local float local_alpha_params[4];
    __local float local_H_X;
    __local float local_H_Y;
    __local float local_H_XY;

    int global_id = get_global_id(0);
    int local_id = get_local_id(0);

    // Load alpha parameters into local memory
    if (local_id < 4) {
        local_alpha_params[local_id] = alpha_params[local_id];
    }

    // Load H_X, H_Y, and H_XY into local memory
    if (local_id == 0) {
        local_H_X = H_X[0];
        local_H_Y = H_Y[0];
        local_H_XY = H_XY[0];
    }

    // Ensure all local memory loads are complete
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute mutual information
    if (global_id == 0) {
        float mutual_information = compute_mutual_information(local_H_X, local_H_Y, local_H_XY, local_alpha_params[0], local_alpha_params[1], local_alpha_params[2], local_alpha_params[3]);
        // Store the result using atomic operation to avoid race conditions
        atomic_xchg(result, mutual_information);
    }
}
