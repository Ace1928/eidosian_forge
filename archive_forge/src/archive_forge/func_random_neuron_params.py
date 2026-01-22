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
def random_neuron_params(simulation_params):
    if not isinstance(simulation_params, SimulationParameters):
        raise TypeError('simulation_params must be an instance of SimulationParameters')
    max_score_range = simulation_params.pattern_params.get('max_score_range', (10, 100))
    return {'threshold': np.clip(np.random.lognormal(mean=np.log(35), sigma=0.5), 0, 127), 'over_threshold': np.clip(np.random.lognormal(mean=np.log(70), sigma=0.5), 0, 127), 'base_signal_strength': np.clip(np.random.lognormal(mean=np.log(64), sigma=0.5), 1, 128), 'refractory_period': np.clip(np.random.lognormal(mean=np.log(3), sigma=0.5), 1, 5), 'spike_threshold': np.clip(np.random.lognormal(mean=np.log(50), sigma=0.5), 1, 127), 'scaling_factor': np.random.uniform(0.1, 2.0), 'neuron_state': random.choice(['excitatory', 'inhibitory', 'neutral']), 'damping_factor': np.clip(np.random.lognormal(mean=np.log(2), sigma=0.3), 1, 5), 'num_sub_windows': np.random.randint(1, 10), 'amplification_factor': np.random.uniform(1, 5), 'temporal_window_size': np.random.randint(5, 20), 'pattern_params': {'max_score': np.random.uniform(*max_score_range)}, 'mapping_params': {'layer1_factor': np.random.uniform(0.1, 1.0), 'layer2_exponent': np.random.uniform(1, 3), 'steepness': np.random.uniform(0.1, 1.0)}, 'output_params': {'expansion_factor': np.random.uniform(1, 5)}}