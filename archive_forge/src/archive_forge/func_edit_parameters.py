import functools
import os
import sys
import traceback
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
import matplotlib.backends.qt_editor.figureoptions as figureoptions
from . import qt_compat
from .qt_compat import (
def edit_parameters(self):
    axes = self.canvas.figure.get_axes()
    if not axes:
        QtWidgets.QMessageBox.warning(self.canvas.parent(), 'Error', 'There are no axes to edit.')
        return
    elif len(axes) == 1:
        ax, = axes
    else:
        titles = [ax.get_label() or ax.get_title() or ax.get_title('left') or ax.get_title('right') or ' - '.join(filter(None, [ax.get_xlabel(), ax.get_ylabel()])) or f'<anonymous {type(ax).__name__}>' for ax in axes]
        duplicate_titles = [title for title in titles if titles.count(title) > 1]
        for i, ax in enumerate(axes):
            if titles[i] in duplicate_titles:
                titles[i] += f' (id: {id(ax):#x})'
        item, ok = QtWidgets.QInputDialog.getItem(self.canvas.parent(), 'Customize', 'Select axes:', titles, 0, False)
        if not ok:
            return
        ax = axes[titles.index(item)]
    figureoptions.figure_edit(ax, self)