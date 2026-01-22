import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import jinja2
import markupsafe
from bs4 import BeautifulSoup
from jupyter_core.paths import jupyter_path
from traitlets import Bool, Unicode, default, validate
from traitlets.config import Config
from jinja2.loaders import split_template_path
from nbformat import NotebookNode
from nbconvert.filters.highlight import Highlight2HTML
from nbconvert.filters.markdown_mistune import IPythonRenderer, MarkdownWithMath
from nbconvert.filters.widgetsdatatypefilter import WidgetsDataTypeFilter
from nbconvert.utils.iso639_1 import iso639_1
from .templateexporter import TemplateExporter
def resources_include_url(name):
    """Get the resources include url for a name."""
    env = self.environment
    mime_type, encoding = mimetypes.guess_type(name)
    try:
        data = env.loader.get_source(env, name)[0].encode('utf8')
    except UnicodeDecodeError:
        pieces = split_template_path(name)
        for searchpath in self.template_paths:
            filename = os.path.join(searchpath, *pieces)
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    data = f.read()
                    break
        else:
            msg = f'No file {name!r} found in {searchpath!r}'
            raise ValueError(msg)
    data = base64.b64encode(data)
    data = data.replace(b'\n', b'').decode('ascii')
    src = f'data:{mime_type};base64,{data}'
    return markupsafe.Markup(src)