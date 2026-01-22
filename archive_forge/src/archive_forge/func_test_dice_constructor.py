import re
import pytest  # NOQA
from .roundtrip import dedent
def test_dice_constructor():
    import srsly.ruamel_yaml
    srsly.ruamel_yaml.add_constructor(u'!dice', dice_constructor)
    with pytest.raises(ValueError):
        data = srsly.ruamel_yaml.load('initial hit points: !dice 8d4', Loader=srsly.ruamel_yaml.Loader)
        assert str(data) == "{'initial hit points': Dice(8,4)}"