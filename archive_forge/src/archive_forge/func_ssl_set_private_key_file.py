import errno
import os
import socket
import sys
import ovs.poller
import ovs.socket_util
import ovs.vlog
@staticmethod
def ssl_set_private_key_file(file_name):
    Stream._SSL_private_key_file = file_name