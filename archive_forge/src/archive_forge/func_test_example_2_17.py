from .roundtrip import YAML
import pytest  # NOQA
@pytest.mark.xfail(strict=True, reason='cannot YAML dump escape sequences (\n) as hex and normal')
def test_example_2_17():
    yaml = YAML()
    yaml.allow_unicode = False
    yaml.preserve_quotes = True
    yaml.round_trip('\n    unicode: "Sosa did fine.\\u263A"\n    control: "\\b1998\\t1999\\t2000\\n"\n    hex esc: "\\x0d\\x0a is \\r\\n"\n\n    single: \'"Howdy!" he cried.\'\n    quoted: \' # Not a \'\'comment\'\'.\'\n    tie-fighter: \'|\\-*-/|\'\n    ')