import os
import testinfra.utils.ansible_runner
def test_redhat_yum_search(host):
    mongodb_version = get_mongodb_version(host)
    if host.system_info.distribution == 'redhat' or host.system_info.distribution == 'centos' or host.system_info.distribution == 'amazon':
        cmd = host.run("yum search mongodb --disablerepo='*'                             --enablerepo='mongodb-{0}'".format(mongodb_version))
        assert cmd.rc == 0
        assert 'MongoDB database server' in cmd.stdout