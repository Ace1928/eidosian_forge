import os
import testinfra.utils.ansible_runner
def test_debian_mongodb_repository_file(host):
    mongodb_version = get_mongodb_version(host)
    if host.system_info.distribution == 'debian' or host.system_info.distribution == 'ubuntu':
        f = host.file('/etc/apt/sources.list.d/mongodb-{0}.list'.format(mongodb_version))
        assert f.exists
        assert f.user == 'root'
        assert f.group == 'root'
        assert f.mode == 420
        assert 'repo.mongodb.org' in f.content_string
        assert mongodb_version in f.content_string