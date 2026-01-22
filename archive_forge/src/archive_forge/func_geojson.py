from plotly.basedatatypes import BaseTraceType as _BaseTraceType
import copy as _copy
@geojson.setter
def geojson(self, val):
    self['geojson'] = val