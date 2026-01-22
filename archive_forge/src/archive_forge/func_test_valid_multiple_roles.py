from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from collections import defaultdict
import json
import os
import subprocess
from gslib.commands import iam
from gslib.exception import CommandException
from gslib.project_id import PopulateProjectId
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import shim_util
from gslib.utils.constants import UTF8
from gslib.utils.iam_helper import BindingsMessageToUpdateDict
from gslib.utils.iam_helper import BindingsDictToUpdateDict
from gslib.utils.iam_helper import BindingStringToTuple as bstt
from gslib.utils.iam_helper import DiffBindings
from gslib.utils.iam_helper import IsEqualBindings
from gslib.utils.iam_helper import PatchBindings
from gslib.utils.retry_util import Retry
from six import add_move, MovedModule
from six.moves import mock
def test_valid_multiple_roles(self):
    """Tests parsing of multiple roles bound to one user."""
    _, bindings = bstt(True, 'allUsers:a,b,c,roles/custom')
    self.assertEqual(len(bindings), 4)
    self.assertIn({'members': ['allUsers'], 'role': 'roles/storage.a'}, bindings)
    self.assertIn({'members': ['allUsers'], 'role': 'roles/storage.b'}, bindings)
    self.assertIn({'members': ['allUsers'], 'role': 'roles/storage.c'}, bindings)
    self.assertIn({'members': ['allUsers'], 'role': 'roles/custom'}, bindings)