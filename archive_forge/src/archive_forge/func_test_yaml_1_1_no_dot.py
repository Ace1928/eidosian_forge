from __future__ import print_function, absolute_import, division, unicode_literals
import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_yaml_1_1_no_dot(self):
    from srsly.ruamel_yaml.error import MantissaNoDotYAML1_1Warning
    with pytest.warns(MantissaNoDotYAML1_1Warning):
        round_trip_load('            %YAML 1.1\n            ---\n            - 1e6\n            ')