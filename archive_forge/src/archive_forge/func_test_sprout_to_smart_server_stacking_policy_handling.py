from breezy.tests.per_repository import TestCaseWithRepository
def test_sprout_to_smart_server_stacking_policy_handling(self):
    """Obey policy where possible, ignore otherwise."""
    stack_on = self.make_branch('stack-on')
    parent_bzrdir = self.make_controldir('.', format='default')
    parent_bzrdir.get_config().set_default_stack_on('stack-on')
    source = self.make_branch('source')
    url = self.make_smart_server('target').abspath('')
    target = source.controldir.sprout(url).open_branch()
    self.assertEqual('../stack-on', target.get_stacked_on_url())
    self.assertEqual(source._format.network_name(), target._format.network_name())