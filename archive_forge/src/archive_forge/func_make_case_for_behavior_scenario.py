from testscenarios import multiply_scenarios
from testtools import TestCase
from testtools.matchers import (
def make_case_for_behavior_scenario(case):
    """Given a test with a behavior scenario installed, make a TestCase."""
    cleanup_behavior = getattr(case, 'cleanup_behavior', None)
    cleanups = [cleanup_behavior] if cleanup_behavior else []
    return make_test_case(case.getUniqueString(), set_up=getattr(case, 'set_up_behavior', _do_nothing), test_body=getattr(case, 'body_behavior', _do_nothing), tear_down=getattr(case, 'tear_down_behavior', _do_nothing), cleanups=cleanups, pre_set_up=getattr(case, 'pre_set_up_behavior', _do_nothing), post_tear_down=getattr(case, 'post_tear_down_behavior', _do_nothing))