import _thread
import errno
import io
import os
import sys
import time
import breezy
from ...lazy_import import lazy_import
import select
import socket
import weakref
from breezy import (
from breezy.i18n import gettext
from breezy.bzr.smart import client, protocol, request, signals, vfs
from breezy.transport import ssh
from ... import errors, osutils
def check_vfs(self, params):
    try:
        request_method = request.request_handlers.get(params.method)
    except KeyError:
        return
    if issubclass(request_method, vfs.VfsRequest):
        raise HpssVfsRequestNotAllowed(params.method, params.args)