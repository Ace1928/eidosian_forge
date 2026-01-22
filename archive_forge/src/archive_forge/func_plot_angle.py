from enum import Enum, auto
from matplotlib import _docstring
def plot_angle(ax, x, y, angle, style):
    phi = np.radians(angle)
    xx = [x + 0.5, x, x + 0.5 * np.cos(phi)]
    yy = [y, y, y + 0.5 * np.sin(phi)]
    ax.plot(xx, yy, lw=12, color='tab:blue', solid_joinstyle=style)
    ax.plot(xx, yy, lw=1, color='black')
    ax.plot(xx[1], yy[1], 'o', color='tab:red', markersize=3)