import unittest
from traits.api import (
from traits.observation.api import (
def test_extended_trait_on_any_value(self):
    team = Team()
    team.any_value = 123
    with self.assertRaises(ValueError) as exception_cm:
        team.observe(lambda e: None, trait('any_value').trait('does_not_exist'))
    self.assertEqual(str(exception_cm.exception), "Trait named 'does_not_exist' not found on 123.")