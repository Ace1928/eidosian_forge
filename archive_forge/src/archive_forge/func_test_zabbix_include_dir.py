import os
import pytest
import testinfra.utils.ansible_runner
def test_zabbix_include_dir(host):
    zabbix_include_dir = host.file('/etc/zabbix/zabbix_proxy.conf.d')
    assert zabbix_include_dir.is_directory
    assert zabbix_include_dir.user == 'zabbix'
    assert zabbix_include_dir.group == 'zabbix'