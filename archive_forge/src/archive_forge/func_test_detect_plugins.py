import importlib.metadata as importlib_metadata
from stevedore import driver
from stevedore import exception
from stevedore import extension
from stevedore.tests import test_extension
from stevedore.tests import utils
def test_detect_plugins(self):
    em = driver.DriverManager('stevedore.test.extension', 't1')
    names = sorted(em.names())
    self.assertEqual(names, ['t1'])