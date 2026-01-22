import os
import pytest
import testinfra.utils.ansible_runner
def test_zabbix_package(host):
    ansible_data = host.ansible.get_variables()
    version = ansible_data['zabbix_proxy_version']
    database = ansible_data['zabbix_proxy_database']
    zabbix_proxy = host.package(f'zabbix-proxy-%s' % database)
    assert str(version) in zabbix_proxy.version