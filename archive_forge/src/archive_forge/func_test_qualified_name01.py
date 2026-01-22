from __future__ import print_function
import sys
import pytest  # NOQA
from .roundtrip import save_and_run  # NOQA
@pytest.mark.skipif(sys.version_info < (3, 0), reason='no __qualname__')
def test_qualified_name01(tmpdir):
    """issue 214"""
    from srsly.ruamel_yaml import YAML
    import srsly.ruamel_yaml.comments
    from srsly.ruamel_yaml.compat import StringIO
    with pytest.raises(ValueError):
        yaml = YAML(typ='unsafe', pure=True)
        yaml.explicit_end = True
        buf = StringIO()
        yaml.dump(srsly.ruamel_yaml.comments.CommentedBase.yaml_anchor, buf)
        res = buf.getvalue()
        assert res == "!!python/name:srsly.ruamel_yaml.comments.CommentedBase.yaml_anchor ''\n...\n"
        x = yaml.load(res)
        assert x == srsly.ruamel_yaml.comments.CommentedBase.yaml_anchor