import functools
import os
import sys
import collections
import importlib
import warnings
from contextvars import copy_context
from importlib.machinery import ModuleSpec
import pkgutil
import threading
import re
import logging
import time
import mimetypes
import hashlib
import base64
import traceback
from urllib.parse import urlparse
from typing import Dict, Optional, Union
import flask
from importlib_metadata import version as _get_distribution_version
from dash import dcc
from dash import html
from dash import dash_table
from .fingerprint import build_fingerprint, check_fingerprint
from .resources import Scripts, Css
from .dependencies import (
from .development.base_component import ComponentRegistry
from .exceptions import (
from .version import __version__
from ._configs import get_combined_config, pathname_configs, pages_folder_config
from ._utils import (
from . import _callback
from . import _get_paths
from . import _dash_renderer
from . import _validate
from . import _watch
from . import _get_app
from ._grouping import map_grouping, grouping_len, update_args_group
from . import _pages
from ._pages import (
from ._jupyter import jupyter_dash, JupyterDisplayMode
from .types import RendererHooks
def strip_relative_path(self, path):
    """
        Return a path with `requests_pathname_prefix` and leading and trailing
        slashes stripped from it. Also, if None is passed in, None is returned.
        Use this function with `get_relative_path` in callbacks that deal
        with `dcc.Location` `pathname` routing.
        That is, your usage may look like:
        ```
        app.layout = html.Div([
            dcc.Location(id='url'),
            html.Div(id='content')
        ])
        @app.callback(Output('content', 'children'), [Input('url', 'pathname')])
        def display_content(path):
            page_name = app.strip_relative_path(path)
            if not page_name:  # None or ''
                return html.Div([
                    dcc.Link(href=app.get_relative_path('/page-1')),
                    dcc.Link(href=app.get_relative_path('/page-2')),
                ])
            elif page_name == 'page-1':
                return chapters.page_1
            if page_name == "page-2":
                return chapters.page_2
        ```
        Note that `chapters.page_1` will be served if the user visits `/page-1`
        _or_ `/page-1/` since `strip_relative_path` removes the trailing slash.

        Also note that `strip_relative_path` is compatible with
        `get_relative_path` in environments where `requests_pathname_prefix` set.
        In some deployment environments, like Dash Enterprise,
        `requests_pathname_prefix` is set to the application name, e.g. `my-dash-app`.
        When working locally, `requests_pathname_prefix` might be unset and
        so a relative URL like `/page-2` can just be `/page-2`.
        However, when the app is deployed to a URL like `/my-dash-app`, then
        `app.get_relative_path('/page-2')` will return `/my-dash-app/page-2`

        The `pathname` property of `dcc.Location` will return '`/my-dash-app/page-2`'
        to the callback.
        In this case, `app.strip_relative_path('/my-dash-app/page-2')`
        will return `'page-2'`

        For nested URLs, slashes are still included:
        `app.strip_relative_path('/page-1/sub-page-1/')` will return
        `page-1/sub-page-1`
        ```
        """
    return _get_paths.app_strip_relative_path(self.config.requests_pathname_prefix, path)