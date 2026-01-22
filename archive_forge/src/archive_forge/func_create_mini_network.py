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
def create_mini_network(simulation_params, connection_mode='specified'):
    if not isinstance(simulation_params, SimulationParameters):
        raise TypeError('simulation_params must be an instance of SimulationParameters')
    neuron_params = [random_neuron_params(simulation_params) for _ in range(7)]
    neurons = [SimplifiedNeuronWithDelay(**params, simulation_params=simulation_params) for params in neuron_params]
    connections = {}
    if connection_mode == 'specified':
        specified_connections = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1), (1, 3), (3, 5), (5, 1), (2, 4), (4, 6), (6, 2), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7)]
        for i, j in specified_connections:
            connections[i, j] = ConnectionWithDelay(**random_connection_params(simulation_params))
    elif connection_mode == 'other_mode':
        pass
    return MiniNetworkWithDelays(neurons, connections)