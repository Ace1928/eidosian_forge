import weakref
from functools import partial
import bokeh
import bokeh.core.properties as bp
import param as pm
from bokeh.model import DataModel
from bokeh.models import ColumnDataSource
from ..reactive import Syncable
from .document import unlocked
from .notebook import push
from .state import state
def cb_bokeh(attr, old, new):
    if attr in _changing:
        return
    try:
        _changing.append(attr)
        obj.param.update(**{attr: new})
    finally:
        _changing.remove(attr)