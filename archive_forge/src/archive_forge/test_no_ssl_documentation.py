import pytest
import urllib3
from dummyserver.testcase import HTTPDummyServerTestCase, HTTPSDummyServerTestCase
from ..test_no_ssl import TestWithoutSSL

Test connections without the builtin ssl module

Note: Import urllib3 inside the test functions to get the importblocker to work
