import string
import pandas._config.config as cf
from pandas.io.formats import printing
def test_repr_obeys_max_seq_limit(self):
    with cf.option_context('display.max_seq_items', 2000):
        assert len(printing.pprint_thing(list(range(1000)))) > 1000
    with cf.option_context('display.max_seq_items', 5):
        assert len(printing.pprint_thing(list(range(1000)))) < 100
    with cf.option_context('display.max_seq_items', 1):
        assert len(printing.pprint_thing(list(range(1000)))) < 9