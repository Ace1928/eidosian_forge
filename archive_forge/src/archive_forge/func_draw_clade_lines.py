import math
import sys
from Bio import MissingPythonDependencyError
def draw_clade_lines(use_linecollection=False, orientation='horizontal', y_here=0, x_start=0, x_here=0, y_bot=0, y_top=0, color='black', lw='.1'):
    """Create a line with or without a line collection object.

        Graphical formatting of the lines representing clades in the plot can be
        customized by altering this function.
        """
    if not use_linecollection and orientation == 'horizontal':
        axes.hlines(y_here, x_start, x_here, color=color, lw=lw)
    elif use_linecollection and orientation == 'horizontal':
        horizontal_linecollections.append(mpcollections.LineCollection([[(x_start, y_here), (x_here, y_here)]], color=color, lw=lw))
    elif not use_linecollection and orientation == 'vertical':
        axes.vlines(x_here, y_bot, y_top, color=color)
    elif use_linecollection and orientation == 'vertical':
        vertical_linecollections.append(mpcollections.LineCollection([[(x_here, y_bot), (x_here, y_top)]], color=color, lw=lw))