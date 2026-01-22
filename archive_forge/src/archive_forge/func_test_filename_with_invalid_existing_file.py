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
def test_filename_with_invalid_existing_file(self):
    filename = os.path.join(self.tmpdir, 'model.pkl')
    with open(filename, 'wb') as pickled_object:
        pickled_object.write(b'this is not a valid pickle')
    model = Model(count=19)
    with mock.patch.object(self.toolkit, 'view_application'):
        with self.assertRaises(pickle.PickleError):
            with self.assertWarns(DeprecationWarning):
                model.configure_traits(filename=filename)