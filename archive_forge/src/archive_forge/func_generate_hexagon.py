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
def generate_hexagon(center: Point3D, side_length: float, elevation: float) -> Hexagon3D:
    """
    Generates the vertices of a 3D hexagon centered at `center`, including the center.
    """
    vertices = [center]
    for i in range(6):
        angle_rad = 2 * math.pi / 6 * i
        x = center[0] + side_length * math.cos(angle_rad)
        y = center[1] + side_length * math.sin(angle_rad)
        vertices.append((x, y, elevation))
    return vertices