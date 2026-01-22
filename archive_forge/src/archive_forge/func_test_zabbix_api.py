import os
import pytest
import testinfra.utils.ansible_runner
def test_zabbix_api(host):
    my_host = host.ansible.get_variables()
    zabbix_api_server_url = str(my_host['zabbix_api_server_url'])
    hostname = 'http://' + zabbix_api_server_url + '/api_jsonrpc.php'
    post_data = '{"jsonrpc": "2.0", "method": "user.login", "params": { "username": "Admin", "password": "zabbix" }, "id": 1, "auth": null}'
    headers = 'Content-Type: application/json-rpc'
    command = "curl -XPOST -H '" + str(headers) + "' -d '" + str(post_data) + "' '" + hostname + "'"
    cmd = host.run(command)
    assert '"jsonrpc":"2.0","result":"' in cmd.stdout