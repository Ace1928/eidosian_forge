from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import pytest  # NOQA
from .roundtrip import round_trip, round_trip_load, round_trip_dump, dedent, YAML
def test_roundtrip_flow_mapping(self):
    import srsly.ruamel_yaml
    s = dedent('        - {a: 1, b: hallo}\n        - {j: fka, k: 42}\n        ')
    data = srsly.ruamel_yaml.load(s, Loader=srsly.ruamel_yaml.RoundTripLoader)
    output = srsly.ruamel_yaml.dump(data, Dumper=srsly.ruamel_yaml.RoundTripDumper)
    assert s == output