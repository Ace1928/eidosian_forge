import importlib.metadata as importlib_metadata
from stevedore import driver
from stevedore import exception
from stevedore import extension
from stevedore.tests import test_extension
from stevedore.tests import utils
def test_multiple_drivers(self):
    extensions = [extension.Extension('backend', importlib_metadata.EntryPoint('backend', 'pkg1:driver', 'backend'), 'pkg backend', None), extension.Extension('backend', importlib_metadata.EntryPoint('backend', 'pkg2:driver', 'backend'), 'pkg backend', None)]
    try:
        dm = driver.DriverManager.make_test_instance(extensions[0])
        dm._init_plugins(extensions)
    except exception.MultipleMatches as err:
        self.assertIn('Multiple', str(err))
    else:
        self.fail('Should have had an error')