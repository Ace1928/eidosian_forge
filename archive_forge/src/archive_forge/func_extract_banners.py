from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.network import NetworkModule, NetworkError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import NetworkConfig, dumps
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.parsing import Command
from ansible_collections.community.network.plugins.module_utils.network.ordnance.ordnance import get_config
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
def extract_banners(config):
    banners = {}
    banner_cmds = re.findall('^banner (\\w+)', config, re.M)
    for cmd in banner_cmds:
        regex = 'banner %s \\^C(.+?)(?=\\^C)' % cmd
        match = re.search(regex, config, re.S)
        if match:
            key = 'banner %s' % cmd
            banners[key] = match.group(1).strip()
    for cmd in banner_cmds:
        regex = 'banner %s \\^C(.+?)(?=\\^C)' % cmd
        match = re.search(regex, config, re.S)
        if match:
            config = config.replace(str(match.group(1)), '')
    config = re.sub('banner \\w+ \\^C\\^C', '!! banner removed', config)
    return (config, banners)