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
def update_x_range(val):
    nonlocal x_axis_range
    x_axis_range[0] = int(val)
    if not dynamic_x_axis[0]:
        ax1.set_xlim(0, x_axis_range[0])
        ax2.set_xlim(0, x_axis_range[0])
        fig.canvas.draw_idle()