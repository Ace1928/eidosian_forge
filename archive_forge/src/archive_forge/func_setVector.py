from reportlab.graphics import shapes
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from weakref import ref as weakref_ref
def setVector(self, **kw):
    for name, value in kw.items():
        for i, v in enumerate(value):
            setattr(self[i], name, v)