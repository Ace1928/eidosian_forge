import asyncio
import base64
import codecs
import datetime as dt
import hashlib
import json
import logging
import os
import re
import urllib.parse as urlparse
import uuid
from base64 import urlsafe_b64encode
from functools import partial
import tornado
from bokeh.server.auth_provider import AuthProvider
from tornado.auth import OAuth2Mixin
from tornado.httpclient import HTTPError as HTTPClientError, HTTPRequest
from tornado.web import HTTPError, RequestHandler, decode_signed_value
from tornado.websocket import WebSocketHandler
from .config import config
from .entry_points import entry_points_for
from .io.resources import (
from .io.state import state
from .util import base64url_encode, decode_token
def set_current_user(self, user):
    if not user:
        self.clear_cookie('is_guest')
        self.clear_cookie('user')
        return
    self.clear_cookie('is_guest')
    self.set_secure_cookie('user', user, expires_days=config.oauth_expiry)
    id_token = base64url_encode(json.dumps({'user': user}))
    if state.encryption:
        id_token = state.encryption.encrypt(id_token.encode('utf-8'))
    self.set_secure_cookie('id_token', id_token, expires_days=config.oauth_expiry)