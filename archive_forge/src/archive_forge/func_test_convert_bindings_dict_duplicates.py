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
def test_convert_bindings_dict_duplicates(self):
    """Test that role and member duplication are converted correctly."""
    expected = defaultdict(set, {'x': set(['y', 'z'])})
    duplicate_roles = [{'role': 'x', 'members': ['y']}, {'role': 'x', 'members': ['z']}]
    duplicate_members = [{'role': 'x', 'members': ['z', 'y']}, {'role': 'x', 'members': ['z']}]
    self.assertEqual(BindingsDictToUpdateDict(duplicate_roles), expected)
    self.assertEqual(BindingsDictToUpdateDict(duplicate_members), expected)