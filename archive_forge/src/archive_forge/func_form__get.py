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
def form__get(self):
    forms = self.forms
    if not forms:
        raise TypeError('You used response.form, but no forms exist')
    if 1 in forms:
        raise TypeError('You used response.form, but more than one form exists')
    return forms[0]