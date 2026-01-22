from __future__ import absolute_import
import os
import sys
import atexit
from sentry_sdk.hub import Hub
from sentry_sdk.utils import logger
from sentry_sdk.integrations import Integration
from sentry_sdk._types import TYPE_CHECKING
def default_callback(pending, timeout):
    """This is the default shutdown callback that is set on the options.
    It prints out a message to stderr that informs the user that some events
    are still pending and the process is waiting for them to flush out.
    """

    def echo(msg):
        sys.stderr.write(msg + '\n')
    echo('Sentry is attempting to send %i pending events' % pending)
    echo('Waiting up to %s seconds' % timeout)
    echo('Press Ctrl-%s to quit' % (os.name == 'nt' and 'Break' or 'C'))
    sys.stderr.flush()