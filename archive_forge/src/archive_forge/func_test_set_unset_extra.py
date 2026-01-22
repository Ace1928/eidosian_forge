from ironicclient.tests.functional.osc.v1 import base
def test_set_unset_extra(self):
    """Check baremetal chassis set and unset commands.

        Test steps:
        1) Create baremetal chassis in setUp.
        2) Set extra data for chassis.
        3) Check that baremetal chassis extra data was set.
        4) Unset extra data for chassis.
        5) Check that baremetal chassis extra data was unset.
        """
    extra_key = 'ext'
    extra_value = 'testdata'
    self.openstack('baremetal chassis set --extra {0}={1} {2}'.format(extra_key, extra_value, self.chassis['uuid']))
    show_prop = self.chassis_show(self.chassis['uuid'], ['extra'])
    self.assertEqual(extra_value, show_prop['extra'][extra_key])
    self.openstack('baremetal chassis unset --extra {0} {1}'.format(extra_key, self.chassis['uuid']))
    show_prop = self.chassis_show(self.chassis['uuid'], ['extra'])
    self.assertNotIn(extra_key, show_prop['extra'])