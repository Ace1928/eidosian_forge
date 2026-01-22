import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import pool as pool
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_pool_set(self):
    new_tls_id, new_ca_id, new_crl_id = ('test-tls-container-id', 'test-ca-tls-container-id', 'test-crl-container-id')
    arglist = [self._po.id, '--name', 'new_name', '--tls-container-ref', new_tls_id, '--ca-tls-container-ref', new_ca_id, '--crl-container-ref', new_crl_id, '--enable-tls', '--tls-ciphers', self._po.tls_ciphers, '--tls-version', self._po.tls_versions[0], '--tls-version', self._po.tls_versions[1], '--alpn-protocol', self._po.alpn_protocols[0], '--alpn-protocol', self._po.alpn_protocols[1]]
    verifylist = [('pool', self._po.id), ('name', 'new_name'), ('tls_ciphers', self._po.tls_ciphers), ('tls_versions', self._po.tls_versions), ('alpn_protocols', self._po.alpn_protocols)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.pool_set.assert_called_with(self._po.id, json={'pool': {'name': 'new_name', 'tls_container_ref': new_tls_id, 'ca_tls_container_ref': new_ca_id, 'crl_container_ref': new_crl_id, 'tls_enabled': True, 'tls_ciphers': self._po.tls_ciphers, 'tls_versions': self._po.tls_versions, 'alpn_protocols': self._po.alpn_protocols}})