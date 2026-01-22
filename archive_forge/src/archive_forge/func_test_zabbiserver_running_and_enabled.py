import os
import pytest
from pathlib import Path
import testinfra.utils.ansible_runner
def test_zabbiserver_running_and_enabled(host):
    zabbix = host.service('zabbix-server')
    if host.system_info.distribution == 'centos':
        assert zabbix.is_enabled
        assert zabbix.is_running
    else:
        assert zabbix.is_running