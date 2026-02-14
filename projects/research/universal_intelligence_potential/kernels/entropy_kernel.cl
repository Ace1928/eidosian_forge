inline void atomic_add_f(volatile __global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal, expected, current;

    current.floatVal = *source;
    do {
        expected.floatVal = current.floatVal;
        newVal.floatVal = expected.floatVal + operand;
        current.intVal = atomic_cmpxchg((volatile __global unsigned int *)source, expected.intVal, newVal.intVal);
    } while (current.intVal != expected.intVal);
}

__kernel void calculate_entropy(
    __global const float *probabilities,
    __global const float *alpha_values,
    __global float *result,
    const int length
) {
    // Initialize local memory for reduction
    __local float local_entropy[256];

    // Get global, local IDs and group size
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_size = get_local_size(0);

    // Read alpha values
    float alpha_H = alpha_values[0];
    float alpha_Pi = alpha_values[1];
    float alpha_log = alpha_values[2];

    // Calculate partial entropy
    float partial_entropy = 0.0f;
    if (global_id < length) {
        float prob = probabilities[global_id];
        if (prob > 0.0f) {
            partial_entropy = -prob * log(prob) / log(alpha_log);
        }
    }
    local_entropy[local_id] = partial_entropy;

    // Perform local reduction
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = group_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            local_entropy[local_id] += local_entropy[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write the result from the first local thread
    if (local_id == 0) {
        atomic_add_f(result, alpha_H * local_entropy[0]);
    }
}
