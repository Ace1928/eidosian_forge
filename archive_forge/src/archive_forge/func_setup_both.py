from .. import tests, utextwrap
def setup_both(testcase, base_class, reused_class):
    super(base_class, testcase).setUp()
    override_textwrap_symbols(testcase)
    reused_class.setUp(testcase)