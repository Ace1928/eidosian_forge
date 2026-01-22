import asyncio
import json
import os
import threading
import warnings
from enum import Enum
from functools import partial
from typing import Awaitable, Dict
import websockets
from jupyterlab_server.themes_handler import ThemesHandler
from markupsafe import Markup
from nbconvert.exporters.html import find_lab_theme
from .static_file_handler import TemplateStaticFileHandler
def create_include_assets_functions(template_name: str, base_url: str) -> Dict:
    return {'include_css': partial(include_css, template_name, base_url), 'include_js': partial(include_js, template_name, base_url), 'include_url': partial(include_url, template_name, base_url), 'include_lab_theme': partial(include_lab_theme, base_url)}