from oslotest import base as test_base
from oslo_utils import dictutils as du
def test_flatten_dict_to_keypairs_with_separator(self):
    data = {'a': 'A', 'b': 'B', 'nested': {'a': 'A', 'b': 'B'}}
    pairs = list(du.flatten_dict_to_keypairs(data, separator='.'))
    self.assertEqual([('a', 'A'), ('b', 'B'), ('nested.a', 'A'), ('nested.b', 'B')], pairs)