import os
import pickle
import pickletools
import shutil
import tempfile
import unittest
import unittest.mock as mock
import warnings
from traits.api import HasTraits, Int
from traits.testing.optional_dependencies import requires_traitsui, traitsui
def test_edit_when_true(self):
    model = Model()
    with mock.patch.object(self.toolkit, 'view_application') as mock_view:
        mock_view.return_value = True
        with self.assertWarns(DeprecationWarning):
            model.configure_traits(edit=True)
    mock_view.assert_called_once()