import re
import pytest  # NOQA
from .roundtrip import dedent
def test_yaml_obj_with_loader_and_dumper():
    import srsly.ruamel_yaml
    srsly.ruamel_yaml.add_representer(Obj1, YAMLObj1.to_yaml, Dumper=srsly.ruamel_yaml.Dumper)
    srsly.ruamel_yaml.add_multi_constructor(YAMLObj1.yaml_tag, YAMLObj1.from_yaml, Loader=srsly.ruamel_yaml.Loader)
    with pytest.raises(ValueError):
        x = srsly.ruamel_yaml.load('!obj:x.2\na: 1', Loader=srsly.ruamel_yaml.Loader)
        print(x)
        assert srsly.ruamel_yaml.dump(x) == '!obj:x.2 "{\'a\': 1}"\n'