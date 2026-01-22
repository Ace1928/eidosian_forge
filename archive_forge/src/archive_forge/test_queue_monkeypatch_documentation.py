from __future__ import absolute_import
import mock
import pytest
from urllib3 import HTTPConnectionPool
from urllib3.exceptions import EmptyPoolError
from urllib3.packages.six.moves import queue

    Test that connection pool works even with a monkey patched Queue module,
    see obspy/obspy#1599, psf/requests#3742, urllib3/urllib3#1061.
    