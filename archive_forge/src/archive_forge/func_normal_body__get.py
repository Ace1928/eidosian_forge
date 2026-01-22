import sys
import io
import random
import mimetypes
import time
import os
import shutil
import smtplib
import shlex
import re
import subprocess
from urllib.parse import urlencode
from urllib import parse as urlparse
from http.cookies import BaseCookie
from paste import wsgilib
from paste import lint
from paste.response import HeaderDict
def normal_body__get(self):
    if self._normal_body is None:
        self._normal_body = self._normal_body_regex.sub(b' ', self.body)
    return self._normal_body