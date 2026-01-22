import logging
import math
import sys
from typing import List, Tuple, Dict, Optional, Union, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
import pandas as pd
from datetime import datetime
def internode_connections(self, hexagon: Hexagon3D, layer: int, ax: plt.Axes, color: str):
    """
        Draws connections from the central node of a hexagon to a vertex in the layer above.
        """
    center = np.mean(np.array(hexagon), axis=0)
    next_layer_hexagons = self.structure[layer + 1]
    for i, next_hexagon in enumerate(next_layer_hexagons):
        target_vertex = next_hexagon[i % 6]
        ax.add_artist(Arrow3D([center[0], target_vertex[0]], [center[1], target_vertex[1]], [center[2], target_vertex[2]], mutation_scale=10, lw=1, arrowstyle='-|>', color=color))