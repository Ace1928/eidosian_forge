import numpy as np
def update_parameters_with_momentum(parameters: dict, grads: dict, v: dict, beta: float, learning_rate: float) -> tuple:
    """
    Update parameters using gradient descent with momentum.

    Args:
        parameters (dict): Dictionary containing your parameters.
        grads (dict): Dictionary containing your gradients for each parameters.
        v (dict): Momentum - moving average of the gradients.
        beta (float): The momentum hyperparameter.
        learning_rate (float): The learning rate.

    Returns:
        tuple: Updated parameters and v.
    """
    L = len(parameters) // 2
    for l in range(L):
        v['dW' + str(l + 1)] = beta * v['dW' + str(l + 1)] + (1 - beta) * grads['dW' + str(l + 1)]
        v['db' + str(l + 1)] = beta * v['db' + str(l + 1)] + (1 - beta) * grads['db' + str(l + 1)]
        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * v['dW' + str(l + 1)]
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * v['db' + str(l + 1)]
    return (parameters, v)