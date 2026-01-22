import datetime
import mock
import pytest
from urllib3.connection import RECENT_DATE, CertificateError, _match_hostname
def test_match_hostname_include_wildcard(self):
    cert = {'subjectAltName': [('DNS', 'foo*')]}
    asserted_hostname = 'foobar'
    _match_hostname(cert, asserted_hostname)