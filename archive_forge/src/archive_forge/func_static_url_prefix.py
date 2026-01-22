from __future__ import annotations
from logging import Logger
from typing import TYPE_CHECKING, Any, cast
from jinja2.exceptions import TemplateNotFound
from jupyter_server.base.handlers import FileFindHandler
@property
def static_url_prefix(self) -> str:
    return self.extensionapp.static_url_prefix