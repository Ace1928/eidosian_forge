from manilaclient.tests.unit import utils
def test_import_v1_share_types_module(self):
    try:
        from manilaclient.v1 import share_types
    except Exception as e:
        msg = "module 'manilaclient.v1.share_types' cannot be imported with error: %s" % str(e)
        assert False, msg
    for cls in ('ShareType', 'ShareTypeManager'):
        msg = "Module 'share_types' has no '%s' attr." % cls
        self.assertTrue(hasattr(share_types, cls), msg)