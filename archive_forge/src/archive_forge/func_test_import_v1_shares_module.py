from manilaclient.tests.unit import utils
def test_import_v1_shares_module(self):
    try:
        from manilaclient.v1 import shares
    except Exception as e:
        msg = "module 'manilaclient.v1.shares' cannot be imported with error: %s" % str(e)
        assert False, msg
    for cls in ('Share', 'ShareManager'):
        msg = "Module 'shares' has no '%s' attr." % cls
        self.assertTrue(hasattr(shares, cls), msg)