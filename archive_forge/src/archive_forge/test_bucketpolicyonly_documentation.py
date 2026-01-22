from __future__ import absolute_import
import re
from gslib.cs_api_map import ApiSelector
import gslib.tests.testcase as testcase
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import unittest
from gslib.utils.retry_util import Retry
Ensures bucketpolicyonly commands fail with too few arguments.