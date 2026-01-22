from typing import List, Tuple, Dict, Callable, Any, Optional
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
import sys
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib import cm  # Corrected import for colormap access
import sys
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
def plot_3d_structure(structure: StructureInfo):
    """
    Plots the 3D structure with arrows to represent connections.
    """
    fig = plt.figure(figsize=(100, 100))
    ax = fig.add_subplot(111, projection='3d')
    color_map = plt.get_cmap('viridis')
    max_layers = max((hexagon[1][0] for hexagon in structure)) + 1
    for hexagon, label in structure:
        layer = label[0]
        color = color_map(layer / float(max_layers))
        hexagon_connections(hexagon, ax, color=color)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.show()