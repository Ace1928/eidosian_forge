import numbers
import prettytable
import yaml
from osc_lib import exceptions as exc
from oslo_serialization import jsonutils
def format_dimensions_query(dims):
    if not dims:
        return {}
    if len(dims) == 1:
        if dims[0].find(';') != -1:
            dims = dims[0].split(';')
        else:
            dims = dims[0].split(',')
    dimensions = {}
    for p in dims:
        try:
            n, v = p.split('=', 1)
        except ValueError:
            n = p
            v = ''
        dimensions[n] = v
    return dimensions