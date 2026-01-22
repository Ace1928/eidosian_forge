import subprocess
import os
import sys
import random
import threading
import collections
import logging
import time
def is_port_in_use(host, port):
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        return sock.connect_ex((host, port)) == 0