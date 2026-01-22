import os
import re
import uuid
from urllib.parse import urlparse
from tornado.escape import url_escape
from ..base.handlers import JupyterHandler
from .decorator import allow_unauthenticated
from .security import passwd_check, set_password
@property
def hashed_password(self):
    return self.password_from_settings(self.settings)