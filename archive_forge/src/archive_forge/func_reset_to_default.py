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
def reset_to_default(event):
    dynamic_x_axis[0] = True
    x_axis_range[0] = 20
    dynamic_check.set_active(0)