import testtools
from ironicclient.v1 import resource_fields
def test_sort_excluded_override_labels(self):
    foo = resource_fields.Resource(['item_3', 'item1', '2nd_item'], sort_excluded=['item1'], override_labels={'item_3': 'Three'})
    self.assertEqual(('Three', 'ITEM1', 'A second item'), foo.labels)
    self.assertEqual(('item_3', '2nd_item'), foo.sort_fields)
    self.assertEqual(('Three', 'A second item'), foo.sort_labels)