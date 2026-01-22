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
def should_probe(self):
    """Should RemoteBzrDirFormat.probe_transport send a smart request on
        this medium?

        Some transports are unambiguously smart-only; there's no need to check
        if the transport is able to carry smart requests, because that's all
        it is for.  In those cases, this method should return False.

        But some HTTP transports can sometimes fail to carry smart requests,
        but still be usuable for accessing remote bzrdirs via plain file
        accesses.  So for those transports, their media should return True here
        so that RemoteBzrDirFormat can determine if it is appropriate for that
        transport.
        """
    return False