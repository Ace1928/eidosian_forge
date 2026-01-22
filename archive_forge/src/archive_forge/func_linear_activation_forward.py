import numpy as np
def linear_activation_forward(A_prev: np.ndarray, W: np.ndarray, b: np.ndarray, activation: str) -> tuple:
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer.

    Args:
        A_prev (np.ndarray): activations from previous layer (or input data).
        W (np.ndarray): weights matrix.
        b (np.ndarray): bias vector.
        activation (str): the activation to be used in this layer, stored as a text string: "sigmoid" or "relu".

    Returns:
        tuple: The activation value from the current layer and the cache.
    """
    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)
    return (A, cache)