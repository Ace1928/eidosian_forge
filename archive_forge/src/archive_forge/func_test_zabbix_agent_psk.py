import os
from zabbix_api import ZabbixAPI
import testinfra.utils.ansible_runner
def test_zabbix_agent_psk(host):
    hostname = host.check_output('hostname -s')
    host_name = 'zabbix-agent-ubuntu'
    psk_file = host.file('/etc/zabbix/zabbix_agent_pskfile.psk')
    if hostname == host_name:
        assert psk_file.user == 'zabbix'
        assert psk_file.group == 'zabbix'
        assert psk_file.mode == 256
        assert psk_file.contains('b7e3d380b9d400676d47198ecf3592ccd4795a59668aa2ade29f0003abbbd40d')
    else:
        assert not psk_file.exists