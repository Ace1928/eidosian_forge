import unittest
def test_suite():
    names = ['test_commands', 'test_dates', 'test_errors', 'test_filter_processor', 'test_info_processor', 'test_helpers', 'test_parser']
    module_names = ['fastimport.tests.' + name for name in names]
    result = unittest.TestSuite()
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(module_names)
    result.addTests(suite)
    return result