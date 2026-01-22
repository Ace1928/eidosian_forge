from novaclient.tests.functional import base
def test_resize_down_revert(self):
    """Tests creating a server and resizes down and reverts the resize.
        Compares quota before, during and after the resize.
        """
    larger_flavor, smaller_flavor = self._create_resize_down_flavors()
    server_id = self._create_server(flavor=larger_flavor).id
    starting_usage = self._get_absolute_limits()
    self.nova('resize', params='%s %s --poll' % (server_id, smaller_flavor))
    resize_usage = self._get_absolute_limits()
    self._compare_quota_usage(starting_usage, resize_usage)
    self.nova('resize-revert', params='%s' % server_id)
    self._wait_for_state_change(server_id, 'active')
    revert_usage = self._get_absolute_limits()
    self._compare_quota_usage(resize_usage, revert_usage)