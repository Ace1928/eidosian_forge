import os
import testinfra.utils.ansible_runner
def test_zabbix_running_and_enabled(host):
    zabbix = host.service('zabbix-java-gateway')
    assert zabbix.is_running