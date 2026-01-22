from .roundtrip import YAML
import pytest  # NOQA
def test_example_2_9():
    yaml = YAML()
    yaml.explicit_start = True
    yaml.indent(sequence=4, offset=2)
    yaml.round_trip('\n    ---\n    hr: # 1998 hr ranking\n      - Mark McGwire\n      - Sammy Sosa\n    rbi:\n      # 1998 rbi ranking\n      - Sammy Sosa\n      - Ken Griffey\n    ')