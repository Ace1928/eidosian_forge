import os
import testinfra.utils.ansible_runner
def test_swappiness_value(host):
    cmd = host.run('cat /proc/sys/vm/swappiness')
    assert cmd.rc == 0
    assert cmd.stdout.strip() == '1'