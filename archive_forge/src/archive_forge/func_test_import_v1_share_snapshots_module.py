from manilaclient.tests.unit import utils
def test_import_v1_share_snapshots_module(self):
    try:
        from manilaclient.v1 import share_snapshots
    except Exception as e:
        msg = "module 'manilaclient.v1.share_snapshots' cannot be imported with error: %s" % str(e)
        assert False, msg
    for cls in ('ShareSnapshot', 'ShareSnapshotManager'):
        msg = "Module 'share_snapshots' has no '%s' attr." % cls
        self.assertTrue(hasattr(share_snapshots, cls), msg)