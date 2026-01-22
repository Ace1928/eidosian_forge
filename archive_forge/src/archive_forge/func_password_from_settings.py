import os
import re
import uuid
from urllib.parse import urlparse
from tornado.escape import url_escape
from ..base.handlers import JupyterHandler
from .decorator import allow_unauthenticated
from .security import passwd_check, set_password
@classmethod
def password_from_settings(cls, settings):
    """DEPRECATED in 2.0, use IdentityProvider API"""
    return settings.get('password', '')