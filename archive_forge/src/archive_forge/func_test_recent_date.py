import datetime
import mock
import pytest
from urllib3.connection import RECENT_DATE, CertificateError, _match_hostname
def test_recent_date(self):
    two_years = datetime.timedelta(days=365 * 2)
    assert RECENT_DATE > (datetime.datetime.today() - two_years).date()