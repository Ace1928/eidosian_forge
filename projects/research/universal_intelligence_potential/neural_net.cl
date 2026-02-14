__kernel void forward_pass(__global float* input, __global float* weights1, __global float* biases1, 
                           __global float* hidden, __global float* weights2, __global float* biases2, 
                           __global float* output, int input_size, int hidden_size, int output_size) {
    int gid = get_global_id(0);

    // Compute hidden layer activations
    if (gid < hidden_size) {
        float sum = biases1[gid];
        for (int i = 0; i < input_size; ++i) {
            sum += input[i] * weights1[i * hidden_size + gid];
        }
        hidden[gid] = tanh(sum);  // Activation function (tanh)
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    // Compute output layer activations
    if (gid < output_size) {
        float sum = biases2[gid];
        for (int i = 0; i < hidden_size; ++i) {
            sum += hidden[i] * weights2[i * output_size + gid];
        }
        output[gid] = sum;  // No activation function for output layer
    }
}
