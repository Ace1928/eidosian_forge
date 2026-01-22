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
def dist_dir(self):
    if self.notebook and self.mode == 'server':
        dist_dir = '/panel-preview/static/extensions/panel/'
    elif self.mode == 'server':
        if state.rel_path:
            dist_dir = f'{state.rel_path}/{LOCAL_DIST}'
        else:
            dist_dir = LOCAL_DIST
        if self.absolute:
            dist_dir = f'{self.root_url}{dist_dir}'
    else:
        dist_dir = CDN_DIST
    return dist_dir