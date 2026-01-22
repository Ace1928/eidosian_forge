from functools import partial
import json
import math
import warnings
from fiona.model import Geometry, to_dict
from fiona._vendor.munch import munchify
def eval_feature_expression(feature, expression):
    safe_dict = {'f': munchify(to_dict(feature))}
    safe_dict.update({'sum': sum, 'pow': pow, 'min': min, 'max': max, 'math': math, 'bool': bool, 'int': partial(nullable, int), 'str': partial(nullable, str), 'float': partial(nullable, float), 'len': partial(nullable, len)})
    try:
        from shapely.geometry import shape
        safe_dict['shape'] = shape
    except ImportError:
        pass
    return eval(expression, {'__builtins__': None}, safe_dict)