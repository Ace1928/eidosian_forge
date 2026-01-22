import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import cos, sin, pi
def draw_hexagon(ax, center, size):
    """
    Draws a single hexagon on the given Matplotlib axis.

    Parameters:
        ax (matplotlib.axes.Axes): The Matplotlib axis to draw on.
        center (tuple): The (x, y) coordinates of the hexagon's center.
        size (float): The radius of the hexagon.
    """
    for i in range(6):
        x1 = center[0] + size * cos(pi / 3 * i)
        y1 = center[1] + size * sin(pi / 3 * i)
        x2 = center[0] + size * cos(pi / 3 * (i + 1))
        y2 = center[1] + size * sin(pi / 3 * (i + 1))
        ax.add_line(plt.Line2D((x1, x2), (y1, y2), color='black'))