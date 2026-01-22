import asyncio
import base64
import time
from collections import defaultdict
import numpy as np
from bokeh.models import (
from panel.io.state import set_curdoc, state
from ...core.options import CallbackError
from ...core.util import datetime_types, dimension_sanitizer, dt64_to_dt, isequal
from ...element import Table
from ...streams import (
from ...util.warnings import warn
from .util import bokeh33, convert_timestamp
@classmethod
def resolve_attr_spec(cls, spec, cb_obj, model=None):
    """
        Resolves a Callback attribute specification looking the
        corresponding attribute up on the cb_obj, which should be a
        bokeh model. If not model is supplied cb_obj is assumed to
        be the same as the model.
        """
    if not cb_obj:
        raise AttributeError(f'Bokeh plot attribute {spec} could not be found')
    if model is None:
        model = cb_obj
    spec = spec.split('.')
    resolved = cb_obj
    for p in spec[1:]:
        if p == 'attributes':
            continue
        if isinstance(resolved, dict):
            resolved = resolved.get(p)
        else:
            resolved = getattr(resolved, p, None)
    return {'id': model.ref['id'], 'value': resolved}