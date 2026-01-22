from . import iter_suite_tests, multiply_scenarios, multiply_tests
def load_tests_apply_scenarios(loader, standard_tests, pattern):
    """Multiply tests depending on their 'scenarios' attribute.

    This can be assigned to 'load_tests' in any test module to make this
    automatically work across tests in the module.
    """
    result = loader.suiteClass()
    multiply_tests_by_their_scenarios(standard_tests, result)
    return result