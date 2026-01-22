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
def showbrowser(self):
    """
        Show this response in a browser window (for debugging purposes,
        when it's hard to read the HTML).
        """
    import webbrowser
    fn = tempnam_no_warning(None, 'paste-fixture') + '.html'
    f = open(fn, 'wb')
    f.write(self.body)
    f.close()
    url = 'file:' + fn.replace(os.sep, '/')
    webbrowser.open_new(url)