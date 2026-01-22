import os
import pytest
import testinfra.utils.ansible_runner
def test_zabbixagent_running_and_enabled(host, zabbix_agent_service):
    if host.system_info.distribution not in ['linuxmint', 'opensuse', 'ubuntu']:
        assert zabbix_agent_service.is_running
        assert zabbix_agent_service.is_enabled