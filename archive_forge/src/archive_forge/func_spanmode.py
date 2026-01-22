from plotly.basedatatypes import BaseTraceType as _BaseTraceType
import copy as _copy
@spanmode.setter
def spanmode(self, val):
    self['spanmode'] = val