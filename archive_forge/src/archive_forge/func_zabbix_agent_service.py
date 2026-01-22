import os
import pytest
import testinfra.utils.ansible_runner
@pytest.fixture
def zabbix_agent_service(host):
    return host.service('zabbix-agent2')