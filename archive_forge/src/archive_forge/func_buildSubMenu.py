import importlib.util
import re
import numpy as np
from .. import colormap
from .. import functions as fn
from ..graphicsItems.GradientPresets import Gradients
from ..Qt import QtCore, QtGui, QtWidgets
def buildSubMenu(self, names, source, sort=True):
    menu = self.sender()
    menu.aboutToShow.disconnect()
    if sort:
        pattern = re.compile('(\\d+)')
        key = lambda x: [int(c) if c.isdigit() else c for c in pattern.split(x)]
        names = sorted(names, key=key)
    for name in names:
        if source == 'preset-gradient':
            cmap = preset_gradient_to_colormap(name)
        else:
            cmap = colormap.get(name, source=source)
        act = QtWidgets.QWidgetAction(menu)
        act.setData((name, source))
        act.setDefaultWidget(buildMenuEntryWidget(cmap, name))
        menu.addAction(act)