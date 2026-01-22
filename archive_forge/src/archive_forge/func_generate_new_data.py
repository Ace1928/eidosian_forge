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
def generate_new_data(existing_data, num_new_points, buffer_size=100):
    """
    Generates new random data for the neural network simulation.

    Parameters:
    existing_data (np.ndarray): Current data buffer of the simulation.
    num_new_points (int): Number of new data points to generate.
    buffer_size (int): Size of the data buffer to maintain.

    Returns:
    np.ndarray: Updated data buffer with new random data appended.
    """
    new_data = np.random.rand(num_new_points, 7)
    combined_data = np.vstack((existing_data, new_data))
    if combined_data.shape[0] > buffer_size:
        return combined_data[-buffer_size:]
    else:
        return combined_data