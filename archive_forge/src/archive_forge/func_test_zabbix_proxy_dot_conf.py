import os
import pytest
import testinfra.utils.ansible_runner
def test_zabbix_proxy_dot_conf(host):
    zabbix_proxy_conf = host.file('/etc/zabbix/zabbix_proxy.conf')
    assert zabbix_proxy_conf.exists
    assert zabbix_proxy_conf.user == 'zabbix'
    assert zabbix_proxy_conf.group == 'zabbix'
    assert zabbix_proxy_conf.mode == 420
    assert zabbix_proxy_conf.contains('ListenPort=10051')
    assert zabbix_proxy_conf.contains('DebugLevel=3')