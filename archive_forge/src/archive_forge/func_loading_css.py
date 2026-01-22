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
@lru_cache(maxsize=None)
def loading_css(loading_spinner, color, max_height):
    return textwrap.dedent(f'\n    :host(.pn-loading):before, .pn-loading:before {{\n      background-color: {color};\n      mask-size: auto calc(min(50%, {max_height}px));\n      -webkit-mask-size: auto calc(min(50%, {max_height}px));\n    }}')