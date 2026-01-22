import os
import testinfra.utils.ansible_runner
def test_limit_file(host):
    f = host.file('/etc/security/limits.conf')
    assert f.exists
    assert f.user == 'root'
    assert f.group == 'root'
    assert f.contains('mongodb\thard\tnproc\t64000')
    assert f.contains('mongodb\thard\tnofile\t64000')
    assert f.contains('mongodb\tsoft\tnproc\t64000')
    assert f.contains('mongodb\tsoft\tnofile\t64000')
    assert f.contains('mongodb\thard\tmemlock\t1024')
    assert f.contains('mongodb\tsoft\tmemlock\t1024')