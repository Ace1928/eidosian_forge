from __future__ import print_function
import httplib2
import json
import os
from select import select
import stat
from sys import stdin
import time
import webbrowser
from base64 import (
from six.moves.urllib.parse import parse_qs
from lazr.restfulclient.errors import HTTPError
from lazr.restfulclient.authorize.oauth import (
from launchpadlib import uris
def notify_end_user_authorization_url(self, authorization_url):
    """Notify the end-user of the URL."""
    super(AuthorizeRequestTokenWithBrowser, self).notify_end_user_authorization_url(authorization_url)
    try:
        browser_obj = webbrowser.get()
        browser = getattr(browser_obj, 'basename', None)
        console_browser = browser in self.TERMINAL_BROWSERS
    except webbrowser.Error:
        browser_obj = None
        console_browser = False
    if console_browser:
        self.output(self.TIMEOUT_MESSAGE % self.TIMEOUT)
        rlist, _, _ = select([stdin], [], [], self.TIMEOUT)
        if rlist:
            stdin.readline()
    if browser_obj is not None:
        webbrowser.open(authorization_url)