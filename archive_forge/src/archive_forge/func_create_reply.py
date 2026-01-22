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
@staticmethod
def create_reply(result, id):
    return Message(Message.T_REPLY, None, None, result, None, id)