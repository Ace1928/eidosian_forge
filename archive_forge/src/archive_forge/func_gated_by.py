import argparse
import fcntl
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from oslo_config import cfg
from oslo_utils import units
from glance.common import config
from glance.i18n import _
def gated_by(predicate):

    def wrap(f):

        def wrapped_f(*args):
            if predicate:
                return f(*args)
            else:
                return None
        return wrapped_f
    return wrap