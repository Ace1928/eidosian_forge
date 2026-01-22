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
def forms__get(self):
    """
        Returns a dictionary of ``Form`` objects.  Indexes are both in
        order (from zero) and by form id (if the form is given an id).
        """
    if self._forms_indexed is None:
        self._parse_forms()
    return self._forms_indexed