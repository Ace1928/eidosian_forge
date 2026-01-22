import datetime
import mock
import pytest
from urllib3.connection import RECENT_DATE, CertificateError, _match_hostname
def test_match_hostname_mismatch(self):
    cert = {'subjectAltName': [('DNS', 'foo')]}
    asserted_hostname = 'bar'
    try:
        with mock.patch('urllib3.connection.log.warning') as mock_log:
            _match_hostname(cert, asserted_hostname)
    except CertificateError as e:
        assert "hostname 'bar' doesn't match 'foo'" in str(e)
        mock_log.assert_called_once_with('Certificate did not match expected hostname: %s. Certificate: %s', 'bar', {'subjectAltName': [('DNS', 'foo')]})
        assert e._peer_cert == cert