from __future__ import annotations
import json
import os
import sys
import uuid
import warnings
from contextlib import contextmanager
from functools import partial
from typing import (
import bokeh
import bokeh.embed.notebook
import param
from bokeh.core.json_encoder import serialize_json
from bokeh.core.templates import MACROS
from bokeh.document import Document
from bokeh.embed import server_document
from bokeh.embed.elements import div_for_render_item, script_for_render_items
from bokeh.embed.util import standalone_docs_json_and_render_items
from bokeh.embed.wrappers import wrap_in_script_tag
from bokeh.models import Model
from bokeh.resources import CDN, INLINE
from bokeh.settings import _Unset, settings
from bokeh.util.serialization import make_id
from param.display import (
from pyviz_comms import (
from ..util import escape
from .embed import embed_state
from .model import add_to_doc, diff
from .resources import (
from .state import state
def ipywidget(obj: Any, doc=None, **kwargs: Any):
    """
    Returns an ipywidget model which renders the Panel object.

    Requires `jupyter_bokeh` to be installed.

    Arguments
    ---------
    obj: object
      Any Panel object or object which can be rendered with Panel
    doc: bokeh.Document
        Bokeh document the bokeh model will be attached to.
    **kwargs: dict
      Keyword arguments passed to the pn.panel utility function

    Returns
    -------
    Returns an ipywidget model which renders the Panel object.
    """
    from jupyter_bokeh.widgets import BokehModel
    from ..pane import panel
    doc = doc if doc else Document()
    model = panel(obj, **kwargs).get_root(doc=doc)
    widget = BokehModel(model, combine_events=True)
    if hasattr(widget, '_view_count'):
        widget._view_count = 0

        def view_count_changed(change, current=[model]):
            new_model = None
            if change['old'] > 0 and change['new'] == 0 and current:
                try:
                    obj._cleanup(current[0])
                except Exception:
                    pass
                current[:] = []
            elif change['old'] == 0 and change['new'] > 0 and (not current or current[0] is not model):
                if current:
                    try:
                        obj._cleanup(current[0])
                    except Exception:
                        pass
                new_model = obj.get_root(doc=widget._document)
                widget.update_from_model(new_model)
                current[:] = [new_model]
        widget.observe(view_count_changed, '_view_count')
    return widget