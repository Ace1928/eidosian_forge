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

__kernel void calculate_adaptability(
    __global const float *adaptation_rates,
    __global const float *alpha_values,
    __global float *result
) {
    // Initialize local memory for reduction
    __local float local_adaptability[256];

    // Get global, local IDs and group size
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_size = get_local_size(0);

    // Read alpha values
    float alpha_A = alpha_values[0];
    float alpha_Adaptation_Rate = alpha_values[1];

    // Calculate partial adaptability
    float partial_adaptability = alpha_A * adaptation_rates[global_id];
    local_adaptability[local_id] = partial_adaptability;

    // Perform local reduction
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = group_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            local_adaptability[local_id] += local_adaptability[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write the result from the first local thread
    if (local_id == 0) {
        atomic_add_f(result, local_adaptability[0]);
    }
}
