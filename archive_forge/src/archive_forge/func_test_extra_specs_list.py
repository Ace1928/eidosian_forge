from cinderclient.tests.functional import base
def test_extra_specs_list(self):
    extra_specs_list = self.cinder('extra-specs-list')
    self.assertTableHeaders(extra_specs_list, ['ID', 'Name', 'extra_specs'])