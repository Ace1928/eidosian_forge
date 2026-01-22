import copy
import stevedore
from glance.common import location_strategy
from glance.common.location_strategy import location_order
from glance.common.location_strategy import store_type
from glance.tests.unit import base
def test_load_strategy_module_with_deduplicating(self):
    modules = ['module1', 'module2']

    def _fake_stevedore_extension_manager(*args, **kwargs):

        def ret():
            return None
        ret.names = lambda: modules
        return ret

    def _fake_stevedore_driver_manager(*args, **kwargs):

        def ret():
            return None
        ret.driver = lambda: None
        ret.driver.__name__ = kwargs['name']
        ret.driver.get_strategy_name = lambda: 'module_name'
        ret.driver.init = lambda: None
        return ret
    self.stub = self.mock_object(stevedore.extension, 'ExtensionManager', _fake_stevedore_extension_manager)
    self.stub = self.mock_object(stevedore.driver, 'DriverManager', _fake_stevedore_driver_manager)
    loaded_modules = location_strategy._load_strategies()
    self.assertEqual(1, len(loaded_modules))
    self.assertIn('module_name', loaded_modules)
    self.assertEqual('module1', loaded_modules['module_name'].__name__)