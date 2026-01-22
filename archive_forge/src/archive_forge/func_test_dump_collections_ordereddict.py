from __future__ import print_function
import sys
import pytest  # NOQA
import platform
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
@pytest.mark.skipif(sys.version_info < (2, 7), reason='collections not available')
def test_dump_collections_ordereddict(self):
    from collections import OrderedDict
    import srsly.ruamel_yaml
    x = OrderedDict([('a', 1), ('b', 2)])
    res = srsly.ruamel_yaml.dump(x, Dumper=srsly.ruamel_yaml.RoundTripDumper, default_flow_style=False)
    assert res == dedent('\n        !!omap\n        - a: 1\n        - b: 2\n        ')