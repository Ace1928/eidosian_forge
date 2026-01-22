import os
import pytest
import testinfra.utils.ansible_runner
@pytest.mark.parametrize('proxysql_package', ['percona-server-client-5.7', 'proxysql'])
def test_proxysql_packages(host, proxysql_package):
    pkg = host.package(proxysql_package)
    assert pkg.is_installed