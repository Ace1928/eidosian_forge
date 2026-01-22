from plotly.basedatatypes import BaseTraceType as _BaseTraceType
import copy as _copy
@boxpoints.setter
def boxpoints(self, val):
    self['boxpoints'] = val