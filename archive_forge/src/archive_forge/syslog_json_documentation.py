from __future__ import (absolute_import, division, print_function)
import logging
import logging.handlers
import socket
from ansible.plugins.callback import CallbackBase

    logs ansible-playbook and ansible runs to a syslog server in json format
    