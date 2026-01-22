import testtools
from ironicclient.v1 import resource_fields
def test_fields_single_value(self):
    foo = resource_fields.Resource(['item1'])
    self.assertEqual(('item1',), foo.fields)
    self.assertEqual(('ITEM1',), foo.labels)
    self.assertEqual(('item1',), foo.sort_fields)
    self.assertEqual(('ITEM1',), foo.sort_labels)