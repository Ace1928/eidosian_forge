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
def patch_stylesheet(stylesheet, dist_url):
    url = stylesheet.url
    if url.startswith(CDN_DIST + dist_url) and dist_url != CDN_DIST:
        patched_url = url.replace(CDN_DIST + dist_url, dist_url)
    elif url.startswith(CDN_DIST) and dist_url != CDN_DIST:
        patched_url = url.replace(CDN_DIST, dist_url)
    elif url.startswith(LOCAL_DIST) and dist_url.lstrip('./').startswith(LOCAL_DIST):
        patched_url = url.replace(LOCAL_DIST, dist_url)
    else:
        return
    version_suffix = f'?v={JS_VERSION}'
    if not patched_url.endswith(version_suffix):
        patched_url += version_suffix
    try:
        stylesheet.url = patched_url
    except Exception:
        pass