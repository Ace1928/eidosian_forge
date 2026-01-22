import os
import pytest
import testinfra.utils.ansible_runner
@pytest.fixture
def zabbix_agent_package(host):
    return host.package('zabbix-agent2')