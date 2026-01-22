from __future__ import annotations
import logging # isort:skip
import json
from tornado.web import authenticated
from .auth_request_handler import AuthRequestHandler
from .session_handler import SessionHandler
 Implements a custom Tornado handler for document display page

    