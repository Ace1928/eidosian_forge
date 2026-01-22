import importlib.metadata as importlib_metadata
from stevedore import driver
from stevedore import exception
from stevedore import extension
from stevedore.tests import test_extension
from stevedore.tests import utils
def test_no_drivers(self):
    try:
        driver.DriverManager('stevedore.test.extension.none', 't1')
    except exception.NoMatches as err:
        self.assertIn("No 'stevedore.test.extension.none' driver found", str(err))