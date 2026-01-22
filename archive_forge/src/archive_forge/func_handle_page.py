from __future__ import annotations
import os
import pickle
import types
from os import path
from typing import Any
from sphinx.application import ENV_PICKLE_FILENAME, Sphinx
from sphinx.builders.html import BuildInfo, StandaloneHTMLBuilder
from sphinx.locale import get_translation
from sphinx.util.osutil import SEP, copyfile, ensuredir, os_path
from sphinxcontrib.serializinghtml import jsonimpl
def handle_page(self, pagename: str, ctx: dict, templatename: str='page.html', outfilename: str | None=None, event_arg: Any=None) -> None:
    ctx['current_page_name'] = pagename
    ctx.setdefault('pathto', lambda p: p)
    self.add_sidebars(pagename, ctx)
    if not outfilename:
        outfilename = path.join(self.outdir, os_path(pagename) + self.out_suffix)
    self.app.emit('html-page-context', pagename, templatename, ctx, event_arg)
    for key in list(ctx):
        if isinstance(ctx[key], types.FunctionType):
            del ctx[key]
    ensuredir(path.dirname(outfilename))
    self.dump_context(ctx, outfilename)
    if ctx.get('sourcename'):
        source_name = path.join(self.outdir, '_sources', os_path(ctx['sourcename']))
        ensuredir(path.dirname(source_name))
        copyfile(self.env.doc2path(pagename), source_name)