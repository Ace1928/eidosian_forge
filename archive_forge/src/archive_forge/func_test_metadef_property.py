import random
import string
from openstack.image.v2 import metadef_namespace as _metadef_namespace
from openstack.image.v2 import metadef_property as _metadef_property
from openstack.tests.functional.image.v2 import base
def test_metadef_property(self):
    metadef_property = self.conn.image.get_metadef_property(self.metadef_property, self.metadef_namespace)
    self.assertIsNotNone(metadef_property)
    self.assertIsInstance(metadef_property, _metadef_property.MetadefProperty)
    self.assertEqual(self.attrs['name'], metadef_property.name)
    self.assertEqual(self.attrs['title'], metadef_property.title)
    self.assertEqual(self.attrs['type'], metadef_property.type)
    self.assertEqual(self.attrs['description'], metadef_property.description)
    self.assertEqual(self.attrs['enum'], metadef_property.enum)
    metadef_properties = list(self.conn.image.metadef_properties(self.metadef_namespace))
    self.assertIsNotNone(metadef_properties)
    self.assertIsInstance(metadef_properties[0], _metadef_property.MetadefProperty)
    self.attrs['title'] = ''.join((random.choice(string.ascii_lowercase) for _ in range(10)))
    self.attrs['description'] = ''.join((random.choice(string.ascii_lowercase) for _ in range(10)))
    metadef_property = self.conn.image.update_metadef_property(self.metadef_property, self.metadef_namespace.namespace, **self.attrs)
    self.assertIsNotNone(metadef_property)
    self.assertIsInstance(metadef_property, _metadef_property.MetadefProperty)
    metadef_property = self.conn.image.get_metadef_property(self.metadef_property.name, self.metadef_namespace)
    self.assertEqual(self.attrs['title'], metadef_property.title)
    self.assertEqual(self.attrs['description'], metadef_property.description)