import copy
from unittest import mock
from oslo_i18n import fixture as i18n_fixture
import suds
from oslo_vmware import exceptions
from oslo_vmware.tests import base
from oslo_vmware import vim
def test_exception_with_deepcopy(self):
    self.assertIsNotNone(self.vim)
    self.assertRaises(exceptions.VimAttributeException, copy.deepcopy, self.vim)