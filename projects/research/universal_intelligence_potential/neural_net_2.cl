__kernel void forward_pass(__global float* input, __global float* weights1, __global float* biases1, 
                           __global float* hidden1, __global float* weights2, __global float* biases2,
                           __global float* hidden2, __global float* weights3, __global float* biases3,
                           __global float* output, int input_size, int hidden_size1, int hidden_size2, int output_size) {
    int gid = get_global_id(0);

    // Compute first hidden layer activations
    if (gid < hidden_size1) {
        float sum = biases1[gid];
        for (int i = 0; i < input_size; ++i) {
            sum += input[i] * weights1[i * hidden_size1 + gid];
        }
        hidden1[gid] = tanh(sum);  // Activation function (tanh)
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    // Compute second hidden layer activations
    if (gid < hidden_size2) {
        float sum = biases2[gid];
        for (int i = 0; i < hidden_size1; ++i) {
            sum += hidden1[i] * weights2[i * hidden_size2 + gid];
        }
        hidden2[gid] = tanh(sum);  // Activation function (tanh)
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    // Compute output layer activations
    if (gid < output_size) {
        float sum = biases3[gid];
        for (int i = 0; i < hidden_size2; ++i) {
            sum += hidden2[i] * weights3[i * output_size + gid];
        }
        output[gid] = sum;  // No activation function for output layer
    }
}
