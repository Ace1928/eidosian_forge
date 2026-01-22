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
def dump_context(self, context: dict, filename: str | os.PathLike[str]) -> None:
    context = context.copy()
    if 'css_files' in context:
        context['css_files'] = [css.filename for css in context['css_files']]
    if 'script_files' in context:
        context['script_files'] = [js.filename for js in context['script_files']]
    if self.implementation_dumps_unicode:
        with open(filename, 'w', encoding='utf-8') as ft:
            self.implementation.dump(context, ft, *self.additional_dump_args)
    else:
        with open(filename, 'wb') as fb:
            self.implementation.dump(context, fb, *self.additional_dump_args)