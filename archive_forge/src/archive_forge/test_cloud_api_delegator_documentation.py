from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib import cloud_api
from gslib import cloud_api_delegator
from gslib import context_config
from gslib import cs_api_map
from gslib.tests import testcase
from gslib.tests.testcase import base
from gslib.tests.util import unittest
from six import add_move, MovedModule
from six.moves import mock
Test delegator class for cloud provider API clients.