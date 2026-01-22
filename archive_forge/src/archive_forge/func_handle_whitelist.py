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
def handle_whitelist(self, digests):
    """Handle the 'whitelist' command in a single step.

        This command increases the ham count for the specified digests."""
    self.server.log.debug('Request to whitelist digests %s', digests)
    if self.server.one_step:
        self.server.database.whitelist(digests)
    else:
        for digest in digests:
            try:
                record = self.server.database[digest]
            except KeyError:
                record = pyzor.engines.common.Record()
            record.wl_increment()
            self.server.database[digest] = record
    if self.server.forwarder:
        for digest in digests:
            self.server.forwarder.queue_forward_request(digest, True)