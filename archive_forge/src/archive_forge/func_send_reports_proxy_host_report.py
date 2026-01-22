from __future__ import (absolute_import, division, print_function)
import os
from datetime import datetime
from collections import defaultdict
import json
import time
from ansible.module_utils._text import to_text
from ansible.module_utils.parsing.convert_bool import boolean as to_bool
from ansible.plugins.callback import CallbackBase
def send_reports_proxy_host_report(self, stats):
    """
        Send reports to Foreman Smart Proxy running Host Reports
        plugin. The format is native Ansible report without any
        changes.
        """
    for host in stats.processed.keys():
        report = {'host': host, 'reported_at': get_now(), 'metrics': {'time': {'total': int(get_time() - self.start_time)}}, 'summary': stats.summarize(host), 'results': self.items[host], 'check_mode': self.check_mode}
        self._send_data('report', 'proxy', host, report)
        self.items[host] = []