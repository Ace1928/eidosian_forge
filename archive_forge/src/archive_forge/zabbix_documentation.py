from __future__ import print_function
import os
import sys
import argparse
import json
import atexit
from ansible.module_utils.six.moves import configparser
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.urls import Request
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError

Zabbix Server external inventory script.
========================================

Returns hosts and hostgroups from Zabbix Server.
If you want to run with --limit against a host group with space in the
name, use asterisk. For example --limit="Linux*servers".

Configuration is read from `zabbix.ini`.

Tested with Zabbix Server 2.0.6, 3.2.3 and 3.4.
