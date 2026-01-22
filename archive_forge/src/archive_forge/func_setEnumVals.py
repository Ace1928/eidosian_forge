from collections import OrderedDict
import numpy as np
from .. import functions as fn
from .. import parametertree as ptree
from ..Qt import QtCore
def setEnumVals(self, opts):
    vals = opts.get('values', {})
    prevState = {}
    for ch in self.children():
        prevState[ch.name()] = ch.value()
        self.removeChild(ch)
    if not isinstance(vals, dict):
        vals = OrderedDict([(v, (str(v), True)) for v in vals])
    for val, valopts in vals.items():
        if isinstance(valopts, bool):
            enabled = valopts
            vname = str(val)
        elif isinstance(valopts, str):
            enabled = True
            vname = valopts
        elif isinstance(valopts, tuple):
            vname, enabled = valopts
        ch = ptree.Parameter.create(name=vname, type='bool', value=prevState.get(vname, enabled))
        ch.maskValue = val
        self.addChild(ch)
    ch = ptree.Parameter.create(name='(other)', type='bool', value=prevState.get('(other)', True))
    ch.maskValue = '__other__'
    self.addChild(ch)