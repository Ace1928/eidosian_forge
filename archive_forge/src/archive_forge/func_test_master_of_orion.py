import pytest
from textwrap import dedent
import platform
import srsly
from .roundtrip import (
def test_master_of_orion(self):
    yaml_str = '\n        - collection: &Backend.Civilizations.RacialPerk\n            items:\n                  - key: perk_population_growth_modifier\n        - *Backend.Civilizations.RacialPerk\n        '
    data = load(yaml_str)