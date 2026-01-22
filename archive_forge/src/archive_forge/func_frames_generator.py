import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import weakref
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.testing.decorators import check_figures_equal
def frames_generator():
    for _ in range(5):
        x = np.linspace(0, 10, 100)
        y = np.random.rand(100)
        frame = Frame(x=x, y=y)
        frames_generated.append(weakref.ref(frame))
        yield frame