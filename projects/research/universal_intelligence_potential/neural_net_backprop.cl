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

__kernel void backward_pass(__global float* input, __global float* weights1, __global float* biases1,
                            __global float* hidden1, __global float* weights2, __global float* biases2,
                            __global float* hidden2, __global float* weights3, __global float* biases3,
                            __global float* output, __global float* target,
                            __global float* weights1_grad, __global float* biases1_grad,
                            __global float* weights2_grad, __global float* biases2_grad,
                            __global float* weights3_grad, __global float* biases3_grad,
                            int input_size, int hidden_size1, int hidden_size2, int output_size, float learning_rate) {
    int gid = get_global_id(0);

    // Compute gradients for output layer
    if (gid < output_size) {
        float error = output[gid] - target[gid];
        for (int i = 0; i < hidden_size2; ++i) {
            atomic_add(&weights3_grad[i * output_size + gid], error * hidden2[i]);
        }
        atomic_add(&biases3_grad[gid], error);
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    // Compute gradients for second hidden layer
    if (gid < hidden_size2) {
        float grad_hidden2 = 0.0f;
        for (int i = 0; i < output_size; ++i) {
            grad_hidden2 += (output[i] - target[i]) * weights3[gid * output_size + i];
        }
        grad_hidden2 *= (1 - hidden2[gid] * hidden2[gid]);  // Derivative of tanh
        for (int i = 0; i < hidden_size1; ++i) {
            atomic_add(&weights2_grad[i * hidden_size2 + gid], grad_hidden2 * hidden1[i]);
        }
        atomic_add(&biases2_grad[gid], grad_hidden2);
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    // Compute gradients for first hidden layer
    if (gid < hidden_size1) {
        float grad_hidden1 = 0.0f;
        for (int i = 0; i < hidden_size2; ++i) {
            float grad_hidden2 = 0.0f;
            for (int j = 0; j < output_size; ++j) {
                grad_hidden2 += (output[j] - target[j]) * weights3[i * output_size + j];
            }
            grad_hidden2 *= (1 - hidden2[i] * hidden2[i]);  // Derivative of tanh
            grad_hidden1 += grad_hidden2 * weights2[gid * hidden_size2 + i];
        }
        grad_hidden1 *= (1 - hidden1[gid] * hidden1[gid]);  // Derivative of tanh
        for (int i = 0; i < input_size; ++i) {
            atomic_add(&weights1_grad[i * hidden_size1 + gid], grad_hidden1 * input[i]);
        }
        atomic_add(&biases1_grad[gid], grad_hidden1);
    }
}

__kernel void update_weights(__global float* weights, __global float* gradients, int size, float learning_rate) {
    int gid = get_global_id(0);
    if (gid < size) {
        weights[gid] -= learning_rate * gradients[gid];
    }
}

__kernel void update_biases(__global float* biases, __global float* gradients, int size, float learning_rate) {
    int gid = get_global_id(0);
    if (gid < size) {
        biases[gid] -= learning_rate * gradients[gid];
    }
}

