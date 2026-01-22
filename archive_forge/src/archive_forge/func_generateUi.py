import numpy as np
from ...Qt import QtCore, QtWidgets
from ...WidgetGroup import WidgetGroup
from ...widgets.ColorButton import ColorButton
from ...widgets.SpinBox import SpinBox
from ..Node import Node
def generateUi(opts):
    """Convenience function for generating common UI types"""
    widget = QtWidgets.QWidget()
    l = QtWidgets.QFormLayout()
    l.setSpacing(0)
    widget.setLayout(l)
    ctrls = {}
    row = 0
    for opt in opts:
        if len(opt) == 2:
            k, t = opt
            o = {}
        elif len(opt) == 3:
            k, t, o = opt
        else:
            raise Exception('Widget specification must be (name, type) or (name, type, {opts})')
        hidden = o.pop('hidden', False)
        tip = o.pop('tip', None)
        if t == 'intSpin':
            w = QtWidgets.QSpinBox()
            if 'max' in o:
                w.setMaximum(o['max'])
            if 'min' in o:
                w.setMinimum(o['min'])
            if 'value' in o:
                w.setValue(o['value'])
        elif t == 'doubleSpin':
            w = QtWidgets.QDoubleSpinBox()
            if 'max' in o:
                w.setMaximum(o['max'])
            if 'min' in o:
                w.setMinimum(o['min'])
            if 'value' in o:
                w.setValue(o['value'])
        elif t == 'spin':
            w = SpinBox()
            w.setOpts(**o)
        elif t == 'check':
            w = QtWidgets.QCheckBox()
            if 'checked' in o:
                w.setChecked(o['checked'])
        elif t == 'combo':
            w = QtWidgets.QComboBox()
            for i in o['values']:
                w.addItem(i)
        elif t == 'color':
            w = ColorButton()
        else:
            raise Exception("Unknown widget type '%s'" % str(t))
        if tip is not None:
            w.setToolTip(tip)
        w.setObjectName(k)
        l.addRow(k, w)
        if hidden:
            w.hide()
            label = l.labelForField(w)
            label.hide()
        ctrls[k] = w
        w.rowNum = row
        row += 1
    group = WidgetGroup(widget)
    return (widget, group, ctrls)