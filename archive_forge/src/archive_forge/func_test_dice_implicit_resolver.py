import re
import pytest  # NOQA
from .roundtrip import dedent
def test_dice_implicit_resolver():
    import srsly.ruamel_yaml
    pattern = re.compile('^\\d+d\\d+$')
    with pytest.raises(ValueError):
        srsly.ruamel_yaml.add_implicit_resolver(u'!dice', pattern)
        assert srsly.ruamel_yaml.dump(dict(treasure=Dice(10, 20)), default_flow_style=False) == 'treasure: 10d20\n'
        assert srsly.ruamel_yaml.load('damage: 5d10', Loader=srsly.ruamel_yaml.Loader) == dict(damage=Dice(5, 10))