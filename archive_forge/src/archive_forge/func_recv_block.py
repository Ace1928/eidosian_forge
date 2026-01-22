import codecs
import errno
import os
import random
import sys
import ovs.json
import ovs.poller
import ovs.reconnect
import ovs.stream
import ovs.timeval
import ovs.util
import ovs.vlog
def recv_block(self):
    while True:
        error, msg = self.recv()
        if error != errno.EAGAIN:
            return (error, msg)
        self.run()
        poller = ovs.poller.Poller()
        self.wait(poller)
        self.recv_wait(poller)
        poller.block()