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

__kernel void calculate_intelligence(
    __global const float *entropies,
    __global const float *mutual_information,
    __global const float *operational_efficiency,
    __global const float *error_management,
    __global const float *adaptability,
    __global const float *volume,
    __global const float *time,
    __global const float *alpha_values,
    __global float *result
) {
    // Initialize local memory for reduction
    __local float local_intelligence[256];

    // Get global, local IDs and group size
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_size = get_local_size(0);

    // Read alpha value
    float k = alpha_values[0];
    
    // Read individual values
    float H_X = entropies[global_id];
    float I_XY = mutual_information[global_id];
    float O = operational_efficiency[global_id];
    float Em = error_management[global_id];
    float A = adaptability[global_id];
    float V = volume[global_id];
    float T = time[global_id];

    // Ensure volume and time are non-zero
    if (V > 0.0f && T > 0.0f) {
        // Calculate partial intelligence
        float partial_intelligence = k * (H_X * I_XY * O * Em * A) / (V * T);
        local_intelligence[local_id] = partial_intelligence;
    } else {
        local_intelligence[local_id] = 0.0f;
    }

    // Perform local reduction
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = group_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            local_intelligence[local_id] += local_intelligence[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write the result from the first local thread
    if (local_id == 0) {
        atomic_add_f(result, local_intelligence[0]);
    }
}
