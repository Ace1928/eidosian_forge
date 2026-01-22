from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy
@customdata.setter
def customdata(self, val):
    self['customdata'] = val