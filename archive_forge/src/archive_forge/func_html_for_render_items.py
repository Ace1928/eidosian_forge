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
def html_for_render_items(docs_json, render_items, template=None, template_variables={}):
    json_id = make_id()
    json = escape(serialize_json(docs_json), quote=False)
    json = wrap_in_script_tag(json, 'application/json', json_id)
    script = wrap_in_script_tag(script_for_render_items(json_id, render_items))
    context = template_variables.copy()
    context.update(dict(title='', plot_script=json + script, docs=render_items, base=NB_TEMPLATE_BASE, macros=MACROS))
    if len(render_items) == 1:
        context['doc'] = context['docs'][0]
        context['roots'] = context['doc'].roots
    if template is None:
        template = NB_TEMPLATE_BASE
    elif isinstance(template, str):
        template = _env.from_string('{% extends base %}\n' + template)
    return template.render(context)