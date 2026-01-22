import os
import testinfra.utils.ansible_runner
def test_mongod_config_default_path(host):
    """
    Ensure that the default paths for RedHat and Debian based OSes are respected
    """
    hostname = host.ansible.get_variables()['inventory_hostname']
    default_path = '/var/lib/mongo'
    if hostname.startswith('centos'):
        default_path = '/var/lib/mongo'
    elif hostname.startswith('ubuntu') or hostname.startswith('debian'):
        default_path = '/var/lib/mongodb'
    f = host.file(default_path)
    assert f.exists
    assert f.is_directory
    conf = host.file('/etc/mongod.conf').content_string
    assert 'dbPath: {0}'.format(default_path) in conf