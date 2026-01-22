from reportlab.graphics import shapes
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from weakref import ref as weakref_ref
def tpcGetItem(obj, x):
    """return obj if it's not a TypedPropertyCollection else obj[x]"""
    return obj[x] if isinstance(obj, TypedPropertyCollection) else obj