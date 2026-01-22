import os
import pytest
import testinfra.utils.ansible_runner
def test_zabbix_proxy_logfile(host):
    zabbix_logfile = host.file('/var/log/zabbix/zabbix_proxy.log')
    assert zabbix_logfile.exists
    assert not zabbix_logfile.contains('Access denied for user')
    assert not zabbix_logfile.contains('database is down: reconnecting')
    assert not zabbix_logfile.contains('Both are missing in the system.')
    assert zabbix_logfile.contains('current database version')
    assert zabbix_logfile.contains('proxy #0 started \\[main process\\]')