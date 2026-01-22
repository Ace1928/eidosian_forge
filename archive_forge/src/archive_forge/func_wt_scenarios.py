from breezy import branchbuilder, tests, transport, workingtree
from breezy.tests import per_controldir, test_server
from breezy.transport import memory
def wt_scenarios():
    """Returns the scenarios for all registered working trees.

    This can used by plugins that want to define tests against these working
    trees.
    """
    scenarios = make_scenarios(tests.default_transport, None, workingtree.format_registry._get_all())
    return scenarios