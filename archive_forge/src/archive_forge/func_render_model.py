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
def render_model(model: 'Model', comm: Optional['Comm']=None, resources: str='cdn') -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    if not isinstance(model, Model):
        raise ValueError('notebook_content expects a single Model instance')
    from ..config import panel_extension as pnext
    target = model.ref['id']
    if not state._is_pyodide and resources == 'server':
        dist_url = '/panel-preview/static/extensions/panel/'
        patch_model_css(model, dist_url=dist_url)
        model.document._template_variables['dist_url'] = dist_url
    docs_json, [render_item] = standalone_docs_json_and_render_items([model], suppress_callback_warning=True)
    div = div_for_render_item(render_item)
    render_json = render_item.to_json()
    requirements = [pnext._globals[ext] for ext in pnext._loaded_extensions if ext in pnext._globals]
    ipywidget = 'ipywidgets_bokeh' in sys.modules
    if not state._is_pyodide:
        ipywidget &= 'PANEL_IPYWIDGET' in os.environ
    script = DOC_NB_JS.render(docs_json=serialize_json(docs_json), render_items=serialize_json([render_json]), requirements=requirements, ipywidget=ipywidget)
    bokeh_script, bokeh_div = (script, div)
    html = "<div id='{id}'>{html}</div>".format(id=target, html=bokeh_div)
    data = {'text/html': html, 'application/javascript': bokeh_script}
    return ({'text/html': mimebundle_to_html(data), EXEC_MIME: ''}, {EXEC_MIME: {'id': target}})