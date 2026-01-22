import re
import pytest  # NOQA
from .roundtrip import dedent
def test_dice_representer():
    import srsly.ruamel_yaml
    srsly.ruamel_yaml.add_representer(Dice, dice_representer)
    assert srsly.ruamel_yaml.dump(dict(gold=Dice(10, 6)), default_flow_style=False) == 'gold: !dice 10d6\n'