import datetime
import enum
import logging
import socket
import sys
import threading
import msgpack
from oslo_privsep._i18n import _
from oslo_utils import uuidutils
def out_of_band(self, msg):
    """Received OOB message. Subclasses might want to override this."""
    pass