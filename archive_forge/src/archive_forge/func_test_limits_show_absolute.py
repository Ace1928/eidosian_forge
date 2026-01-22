from manilaclient.tests.functional.osc import base
def test_limits_show_absolute(self):
    limits = self.listing_result('share', ' limits show --absolute')
    self.assertTableStruct(limits, ['Name', 'Value'])