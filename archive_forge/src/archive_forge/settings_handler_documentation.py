from __future__ import annotations
import json
from typing import Any
from jsonschema import ValidationError
from jupyter_server.extension.handler import ExtensionHandlerJinjaMixin, ExtensionHandlerMixin
from tornado import web
from .settings_utils import SchemaHandler, get_settings, save_settings
from .translation_utils import translator
Update a setting