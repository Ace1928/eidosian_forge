import unittest
from traits.api import (
from traits.observation.api import (
def test_no_new_trait_added(self):
    team = Team()
    team.observe(lambda e: None, trait('leader').trait('does_not_exist'))
    with self.assertRaises(ValueError):
        team.leader = Person()
    self.assertNotIn('does_not_exist', team.leader.trait_names())