import base64
import inspect
import logging
import socket
import subprocess
import uuid
from contextlib import closing
from itertools import islice
from sys import version_info
def check_port_connectivity():
    port = find_free_port()
    try:
        with subprocess.Popen(['nc', '-l', '-p', str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as server:
            with subprocess.Popen(['nc', '-zv', 'localhost', str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as client:
                client.wait()
                server.terminate()
                return client.returncode == 0
    except Exception as e:
        _logger.warning('Failed to check port connectivity: %s', e)
        return False