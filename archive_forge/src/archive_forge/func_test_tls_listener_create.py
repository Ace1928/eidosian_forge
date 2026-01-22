import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import listener
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
@mock.patch('octaviaclient.osc.v2.utils.get_listener_attrs')
def test_tls_listener_create(self, mock_client):
    mock_client.return_value = self.listener_info
    arglist = ['mock_lb_id', '--name', self._listener.name, '--protocol', 'TERMINATED_HTTPS'.lower(), '--protocol-port', '443', '--sni-container-refs', self._listener.sni_container_refs[0], self._listener.sni_container_refs[1], '--default-tls-container-ref', self._listener.default_tls_container_ref, '--client-ca-tls-container-ref', self._listener.client_ca_tls_container_ref, '--client-authentication', self._listener.client_authentication, '--client-crl-container-ref', self._listener.client_crl_container_ref, '--tls-ciphers', self._listener.tls_ciphers, '--tls-version', self._listener.tls_versions[0], '--tls-version', self._listener.tls_versions[1], '--alpn-protocol', self._listener.alpn_protocols[0], '--alpn-protocol', self._listener.alpn_protocols[1], '--hsts-max-age', '12000000', '--hsts-include-subdomains', '--hsts-preload']
    verifylist = [('loadbalancer', 'mock_lb_id'), ('name', self._listener.name), ('protocol', 'TERMINATED_HTTPS'), ('protocol_port', 443), ('sni_container_refs', self._listener.sni_container_refs), ('default_tls_container_ref', self._listener.default_tls_container_ref), ('client_ca_tls_container_ref', self._listener.client_ca_tls_container_ref), ('client_authentication', self._listener.client_authentication), ('client_crl_container_ref', self._listener.client_crl_container_ref), ('tls_ciphers', self._listener.tls_ciphers), ('tls_versions', self._listener.tls_versions), ('alpn_protocols', self._listener.alpn_protocols), ('hsts_max_age', 12000000), ('hsts_include_subdomains', True), ('hsts_preload', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.listener_create.assert_called_with(json={'listener': self.listener_info})