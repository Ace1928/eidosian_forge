from .. import PlotItem
from .. import functions as fn
from ..Qt import QtCore, QtWidgets
from .Exporter import Exporter
def cleanAxes(self, axl):
    if type(axl) is not list:
        axl = [axl]
    for ax in axl:
        if ax is None:
            continue
        for loc, spine in ax.spines.items():
            if loc in ['left', 'bottom']:
                pass
            elif loc in ['right', 'top']:
                spine.set_color('none')
            else:
                raise ValueError('Unknown spine location: %s' % loc)
            ax.xaxis.set_ticks_position('bottom')