import datetime
import mock
import pytest
from urllib3.connection import RECENT_DATE, CertificateError, _match_hostname
def test_match_hostname_ip_address_ipv6_brackets(self):
    cert = {'subjectAltName': (('IP Address', '1:2::2:1'),)}
    asserted_hostname = '[1:2::2:1]'
    _match_hostname(cert, asserted_hostname)