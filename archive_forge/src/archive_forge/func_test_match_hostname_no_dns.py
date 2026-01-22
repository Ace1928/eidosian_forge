import datetime
import mock
import pytest
from urllib3.connection import RECENT_DATE, CertificateError, _match_hostname
def test_match_hostname_no_dns(self):
    cert = {'subjectAltName': [('DNS', '')]}
    asserted_hostname = 'bar'
    try:
        with mock.patch('urllib3.connection.log.warning') as mock_log:
            _match_hostname(cert, asserted_hostname)
    except CertificateError as e:
        assert "hostname 'bar' doesn't match ''" in str(e)
        mock_log.assert_called_once_with('Certificate did not match expected hostname: %s. Certificate: %s', 'bar', {'subjectAltName': [('DNS', '')]})
        assert e._peer_cert == cert