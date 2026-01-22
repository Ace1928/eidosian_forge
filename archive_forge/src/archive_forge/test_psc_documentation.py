from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from boto import config
from gslib.gcs_json_api import DEFAULT_HOST
from gslib.tests import testcase
from gslib.tests.testcase import integration_testcase
from gslib.tests.util import ObjectToURI
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import unittest
Integration tests for PSC custom endpoints.