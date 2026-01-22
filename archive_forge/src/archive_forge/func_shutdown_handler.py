import os
import sys
import time
import errno
import socket
import signal
import logging
import threading
import traceback
import email.message
import pyzor.config
import pyzor.account
import pyzor.engines.common
import pyzor.hacks.py26
def shutdown_handler(self, *args, **kwargs):
    """Handler for the SIGTERM signal. This should be used to kill the
        daemon and ensure proper clean-up.
        """
    self.log.info('SIGTERM received. Shutting down.')
    t = threading.Thread(target=self.shutdown)
    t.start()