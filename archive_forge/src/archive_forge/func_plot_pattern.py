from math import exp, pi, sin, sqrt, cos, acos
import numpy as np
from ase.data import atomic_numbers
def plot_pattern(self, filename=None, show=False, ax=None):
    """ Plot XRD or SAXS depending on filled data

        Uses Matplotlib to plot pattern. Use *show=True* to
        show the figure and *filename='abc.png'* or
        *filename='abc.eps'* to save the figure to a file.

        Returns:
            ``matplotlib.axes.Axes`` object."""
    import matplotlib.pyplot as plt
    if ax is None:
        plt.clf()
        ax = plt.gca()
    if self.mode == 'XRD':
        x, y = (np.array(self.twotheta_list), np.array(self.intensity_list))
        ax.plot(x, y / np.max(y), '.-')
        ax.set_xlabel('2$\\theta$')
        ax.set_ylabel('Intensity')
    elif self.mode == 'SAXS':
        x, y = (np.array(self.q_list), np.array(self.intensity_list))
        ax.loglog(x, y / np.max(y), '.-')
        ax.set_xlabel('q, 1/Angstr.')
        ax.set_ylabel('Intensity')
    else:
        raise Exception('No data available, call calc_pattern() first')
    if show:
        plt.show()
    if filename is not None:
        fig = ax.get_figure()
        fig.savefig(filename)
    return ax