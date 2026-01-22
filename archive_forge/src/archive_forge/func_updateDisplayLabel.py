import numpy as np
from ...Qt import QtCore, QtWidgets
from ..Parameter import Parameter
from .basetypes import Emitter, WidgetParameterItem
def updateDisplayLabel(self, value=None):
    if value is None:
        value = self.param.value()
    self.sliderLabel.setText(self.prettyTextValue(self.slider.value()))
    value = str(value)
    if self._suffix is None:
        suffixTxt = ''
    else:
        suffixTxt = f' {self._suffix}'
    self.displayLabel.setText(value + suffixTxt)