from __future__ import print_function
import sys
import pytest  # NOQA
from .roundtrip import save_and_run  # NOQA
def test_monster(tmpdir):
    program_src = u'    import srsly.ruamel_yaml\n    from textwrap import dedent\n\n    class Monster(srsly.ruamel_yaml.YAMLObject):\n        yaml_tag = u\'!Monster\'\n\n        def __init__(self, name, hp, ac, attacks):\n            self.name = name\n            self.hp = hp\n            self.ac = ac\n            self.attacks = attacks\n\n        def __repr__(self):\n            return "%s(name=%r, hp=%r, ac=%r, attacks=%r)" % (\n                self.__class__.__name__, self.name, self.hp, self.ac, self.attacks)\n\n    data = srsly.ruamel_yaml.load(dedent("""\\\n        --- !Monster\n        name: Cave spider\n        hp: [2,6]    # 2d6\n        ac: 16\n        attacks: [BITE, HURT]\n    """), Loader=srsly.ruamel_yaml.Loader)\n    # normal dump, keys will be sorted\n    assert srsly.ruamel_yaml.dump(data) == dedent("""\\\n        !Monster\n        ac: 16\n        attacks: [BITE, HURT]\n        hp: [2, 6]\n        name: Cave spider\n    """)\n    '
    assert save_and_run(program_src, tmpdir) == 1