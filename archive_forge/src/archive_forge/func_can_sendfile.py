import io
import logging
import os
import re
import sys
from gunicorn.http.message import HEADER_RE
from gunicorn.http.errors import InvalidHeader, InvalidHeaderName
from gunicorn import SERVER_SOFTWARE, SERVER
from gunicorn import util
def can_sendfile(self):
    return self.cfg.sendfile is not False