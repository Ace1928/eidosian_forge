from plotly.basedatatypes import BaseLayoutType as _BaseLayoutType
import copy as _copy
@shapedefaults.setter
def shapedefaults(self, val):
    self['shapedefaults'] = val