import httplib
import pytest
import StringIO
from mock import patch
from ..test_no_ssl import TestWithoutSSL

        Check that URLFetch is used when fetching https resources
        