from __future__ import (absolute_import, division, print_function)
import glob
import os
import re
import socket
import struct
from ansible.module_utils.facts.network.base import Network, NetworkCollector
from ansible.module_utils.facts.utils import get_file_content
def get_ethtool_data(self, device):
    data = {}
    ethtool_path = self.module.get_bin_path('ethtool')
    if ethtool_path:
        args = [ethtool_path, '-k', device]
        rc, stdout, stderr = self.module.run_command(args, errors='surrogate_then_replace')
        if rc == 0:
            features = {}
            for line in stdout.strip().splitlines():
                if not line or line.endswith(':'):
                    continue
                key, value = line.split(': ')
                if not value:
                    continue
                features[key.strip().replace('-', '_')] = value.strip()
            data['features'] = features
        args = [ethtool_path, '-T', device]
        rc, stdout, stderr = self.module.run_command(args, errors='surrogate_then_replace')
        if rc == 0:
            data['timestamping'] = [m.lower() for m in re.findall('SOF_TIMESTAMPING_(\\w+)', stdout)]
            data['hw_timestamp_filters'] = [m.lower() for m in re.findall('HWTSTAMP_FILTER_(\\w+)', stdout)]
            m = re.search('PTP Hardware Clock: (\\d+)', stdout)
            if m:
                data['phc_index'] = int(m.groups()[0])
    return data