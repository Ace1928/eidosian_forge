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

__kernel void calculate_operational_efficiency(
    __global const float *performance,
    __global const float *energy,
    __global const float *alpha_values,
    __global float *result
) {
    // Initialize local memory for reduction
    __local float local_efficiency[256];

    // Get global, local IDs and group size
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_size = get_local_size(0);

    // Read alpha values
    float alpha_O = alpha_values[0];
    float alpha_P = alpha_values[1];
    float alpha_E = alpha_values[2];

    // Calculate partial operational efficiency
    float partial_efficiency = 0.0f;
    if (energy[global_id] > 0.0f) {
        partial_efficiency = alpha_O * (performance[global_id] / energy[global_id]);
    }
    local_efficiency[local_id] = partial_efficiency;

    // Perform local reduction
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = group_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            local_efficiency[local_id] += local_efficiency[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write the result from the first local thread
    if (local_id == 0) {
        atomic_add_f(result, local_efficiency[0]);
    }
}
