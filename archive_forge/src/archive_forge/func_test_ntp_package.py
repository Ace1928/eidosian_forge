import os
import testinfra.utils.ansible_runner
def test_ntp_package(host):
    ntp = host.package('ntp')
    chrony = host.package('chrony')
    ntpsec = host.package('ntpsec')
    assert ntp.is_installed or chrony.is_installed or ntpsec.is_installed