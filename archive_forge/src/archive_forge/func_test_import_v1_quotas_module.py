from manilaclient.tests.unit import utils
def test_import_v1_quotas_module(self):
    try:
        from manilaclient.v1 import quotas
    except Exception as e:
        msg = "module 'manilaclient.v1.quotas' cannot be imported with error: %s" % str(e)
        assert False, msg
    for cls in ('QuotaSet', 'QuotaSetManager'):
        msg = "Module 'quotas' has no '%s' attr." % cls
        self.assertTrue(hasattr(quotas, cls), msg)