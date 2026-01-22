import ast
import base64
import logging
import os
import pathlib
from glob import glob
from types import ModuleType
from bokeh.application import Application
from bokeh.application.handlers.document_lifecycle import (
from bokeh.application.handlers.function import FunctionHandler
from bokeh.command.subcommands.serve import Serve as _BkServe
from bokeh.command.util import build_single_handler_applications
from bokeh.core.validation import silence
from bokeh.core.validation.warnings import EMPTY_LAYOUT
from bokeh.server.contexts import ApplicationContext
from tornado.ioloop import PeriodicCallback
from tornado.web import StaticFileHandler
from ..auth import BasicAuthProvider, OAuthProvider
from ..config import config
from ..io.document import _cleanup_doc
from ..io.liveness import LivenessHandler
from ..io.reload import record_modules, watch
from ..io.rest import REST_PROVIDERS
from ..io.server import INDEX_HTML, get_static_routes, set_curdoc
from ..io.state import state
from ..util import fullpath
def parse_vars(items):
    """
    Parse a series of key-value pairs and return a dictionary
    """
    return dict((parse_var(item) for item in items))