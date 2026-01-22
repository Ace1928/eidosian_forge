import os
import pytest
import testinfra.utils.ansible_runner
@pytest.fixture
def zabbix_agent_include_dir(host):
    return host.file('/etc/zabbix/zabbix_agent2.d')