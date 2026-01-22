from __future__ import annotations
from sphinx.util import logging  # isort:skip
from os.path import basename
from docutils import nodes
from sphinx.locale import _
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective
from .util import get_sphinx_resources
def purge_gallery_xrefs(app, env, docname):
    if not hasattr(env, 'all_gallery_overview'):
        return
    env.all_gallery_overview = [xref for xref in env.all_gallery_overview if xref['docname'] != docname]