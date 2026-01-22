from __future__ import (absolute_import, division, print_function)
import os
from datetime import datetime
from collections import defaultdict
import json
import time
from ansible.module_utils._text import to_text
from ansible.module_utils.parsing.convert_bool import boolean as to_bool
from ansible.plugins.callback import CallbackBase
def send_facts(self):
    """
        Sends facts to Foreman, to be parsed by foreman_ansible fact
        parser.  The default fact importer should import these facts
        properly.
        """
    if self.report_type == 'proxy':
        return
    for host, facts in self.facts.items():
        facts = {'name': host, 'facts': {'ansible_facts': facts, '_type': 'ansible', '_timestamp': get_now()}}
        self._send_data('facts', 'foreman', host, facts)