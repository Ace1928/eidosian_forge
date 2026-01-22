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
def list_param_to_ppt(p, kwargs):
    if isinstance(p.item_type, type) and issubclass(p.item_type, pm.Parameterized):
        return (bp.List(bp.Instance(DataModel)), [(ParameterizedList, lambda ps: [create_linked_datamodel(p) for p in ps])])
    return bp.List(bp.Any, **kwargs)