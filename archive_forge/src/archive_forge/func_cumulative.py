from plotly.basedatatypes import BaseTraceType as _BaseTraceType
import copy as _copy
@cumulative.setter
def cumulative(self, val):
    self['cumulative'] = val