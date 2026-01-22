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
def interactive_cli():
    """
    Interactive CLI for generating and visualizing the hexagonal structure.
    """
    try:
        neurons = int(input('Enter the number of neurons for the first layer: '))
        side_length = 1.0
        elevation = 0.0
        center = (0.0, 0.0, elevation)
        hexagon = generate_hexagon(center, side_length, elevation)
        structure = [(hexagon, [0, 0, 0, 0, 0, 0])]
        plot_3d_structure(structure)
    except ValueError:
        print('Please enter a valid integer for the number of neurons.')
        sys.exit(1)