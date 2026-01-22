import testtools
from ironicclient.v1 import resource_fields
def test_fields_multiple_value_order(self):
    foo = resource_fields.Resource(['2nd_item', 'item1'])
    self.assertEqual(('2nd_item', 'item1'), foo.fields)
    self.assertEqual(('A second item', 'ITEM1'), foo.labels)
    self.assertEqual(('2nd_item', 'item1'), foo.sort_fields)
    self.assertEqual(('A second item', 'ITEM1'), foo.sort_labels)