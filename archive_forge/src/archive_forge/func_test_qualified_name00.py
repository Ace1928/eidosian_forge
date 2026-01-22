from __future__ import print_function
import sys
import pytest  # NOQA
from .roundtrip import save_and_run  # NOQA
@pytest.mark.skipif(sys.version_info < (3, 0), reason='no __qualname__')
def test_qualified_name00(tmpdir):
    """issue 214"""
    program_src = u'    from srsly.ruamel_yaml import YAML\n    from srsly.ruamel_yaml.compat import StringIO\n\n    class A:\n        def f(self):\n            pass\n\n    yaml = YAML(typ=\'unsafe\', pure=True)\n    yaml.explicit_end = True\n    buf = StringIO()\n    yaml.dump(A.f, buf)\n    res = buf.getvalue()\n    print(\'res\', repr(res))\n    assert res == "!!python/name:__main__.A.f \'\'\\n...\\n"\n    x = yaml.load(res)\n    assert x == A.f\n    '
    assert save_and_run(program_src, tmpdir) == 1