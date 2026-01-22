from unittest import mock
from magnumclient.common import cliutils
from magnumclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.tests.v1 import test_clustertemplates_shell
from magnumclient.v1.clusters import Cluster
def test_cluster_config_k8s_success_without_tls_non_csh(self):
    self._test_cluster_config_success(coe='kubernetes', shell='zsh', tls_disable=True)