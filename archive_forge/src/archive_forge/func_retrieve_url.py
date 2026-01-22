import argparse
import io
import logging
import os
import platform
import re
import shutil
import sys
import subprocess
from . import envvar
from .deprecation import deprecated
from .errors import DeveloperError
import pyomo.common
from pyomo.common.dependencies import attempt_import
def retrieve_url(self, url):
    """Return the contents of a URL as an io.BytesIO object"""
    ctx = ssl.create_default_context()
    if self.cacert:
        ctx.load_verify_locations(cafile=self.cacert)
    if self.insecure:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    try:
        fetch = request.urlopen(url, context=ctx)
    except urllib_error.HTTPError as e:
        if e.code != 403:
            raise
        fetch = None
    if fetch is None:
        req = request.Request(url=url, headers={'User-Agent': 'Mozilla/5.0'})
        fetch = request.urlopen(req, context=ctx)
    ans = fetch.read()
    logger.info('  ...downloaded %s bytes' % (len(ans),))
    return ans