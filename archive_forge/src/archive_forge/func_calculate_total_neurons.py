import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from collections import deque
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox
from matplotlib.animation import FuncAnimation
import logging
import datetime
import sys
import cProfile
def calculate_total_neurons(N, W):
    """
    Calculates the total number of neurons in a layered neural network.

    Parameters:
    N (int): Number of layers in the network.
    W (int): Width of each layer.

    Returns:
    int: Total number of neurons in the network.
    """
    if not isinstance(N, int) or N <= 0:
        raise ValueError('N must be a positive integer')
    if not isinstance(W, int) or W <= 0:
        raise ValueError('W must be a positive integer')
    neurons_base_layer = W * 7 * 6 ** (N - 1)
    neurons_above_base = W * sum((6 ** (N - i) for i in range(2, N + 1)))
    total_neurons = neurons_base_layer + neurons_above_base
    return total_neurons