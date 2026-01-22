import contextlib
import re
import xml.dom.minidom as xml
import numpy as np
from .. import debug
from .. import functions as fn
from ..parametertree import Parameter
from ..Qt import QtCore, QtGui, QtSvg, QtWidgets
from .Exporter import Exporter
def widthChanged(self):
    sr = self.getSourceRect()
    ar = sr.height() / sr.width()
    self.params.param('height').setValue(self.params['width'] * ar, blockSignal=self.heightChanged)