from __future__ import annotations
import functools
import importlib
import json
import logging
import mimetypes
import os
import pathlib
import re
import textwrap
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import (
import param
from bokeh.embed.bundle import (
from bokeh.model import Model
from bokeh.models import ImportedStyleSheet
from bokeh.resources import Resources as BkResources, _get_server_urls
from bokeh.settings import settings as _settings
from jinja2.environment import Environment
from jinja2.loaders import FileSystemLoader
from markupsafe import Markup
from ..config import config, panel_extension as extension
from ..util import isurl, url_path
from .state import state
@property
def css_raw(self):
    from ..config import config
    raw = super(Resources, self).css_raw
    css_files = self._collect_external_resources('__css__')
    self.extra_resources(css_files, '__css__')
    if self.mode.lower() not in ('server', 'cdn'):
        raw += [(DIST_DIR / css.replace(CDN_DIST, '')).read_text(encoding='utf-8') for css in css_files if is_cdn_url(css)]
    for cssf in config.css_files:
        if not os.path.isfile(cssf):
            continue
        css_txt = process_raw_css([Path(cssf).read_text(encoding='utf-8')])[0]
        if css_txt not in raw:
            raw.append(css_txt)
    if config.global_loading_spinner:
        loading_base = (DIST_DIR / 'css' / 'loading.css').read_text(encoding='utf-8')
        raw.extend([loading_base, loading_css(config.loading_spinner, config.loading_color, config.loading_max_height)])
    return raw + process_raw_css(config.raw_css) + process_raw_css(config.global_css)