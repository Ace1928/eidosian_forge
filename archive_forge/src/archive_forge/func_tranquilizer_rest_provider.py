import json
import os
import tempfile
import traceback
from runpy import run_path
from unittest.mock import MagicMock
from urllib.parse import parse_qs
import param
from tornado import web
from tornado.wsgi import WSGIContainer
from ..entry_points import entry_points_for
from .state import state
def tranquilizer_rest_provider(files, endpoint):
    """
    Returns a Tranquilizer based REST API. Builds the API by evaluating
    the scripts and notebooks being served and finding all tranquilized
    functions inside them.

    Arguments
    ---------
    files: list(str)
      A list of paths being served
    endpoint: str
      The endpoint to serve the REST API on

    Returns
    -------
    A Tornado routing pattern containing the route and handler
    """
    app = build_tranquilize_application(files)
    tr = WSGIContainer(app)
    return [('^/%s/.*' % endpoint, web.FallbackHandler, dict(fallback=tr))]