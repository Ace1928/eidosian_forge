import os
import testinfra.utils.ansible_runner
def test_redhat_mongodb_repository_file(host):
    mongodb_version = get_mongodb_version(host)
    if host.system_info.distribution == 'redhat' or host.system_info.distribution == 'centos' or host.system_info.distribution == 'amazon':
        f = host.file('/etc/yum.repos.d/mongodb-{0}.repo'.format(mongodb_version))
        assert f.exists
        assert f.user == 'root'
        assert f.group == 'root'
        assert f.mode == 420
        assert f.md5sum == 'dbcb01e2e25b6d10afd27b60205136c3'