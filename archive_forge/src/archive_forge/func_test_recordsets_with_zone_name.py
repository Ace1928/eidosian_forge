import random
import string
from testtools import content
from openstack.tests.functional import base
def test_recordsets_with_zone_name(self):
    """Test DNS recordsets functionality"""
    sub = ''.join((random.choice(string.ascii_lowercase) for _ in range(6)))
    zone = '%s.example2.net.' % sub
    email = 'test@example2.net'
    name = 'www.%s' % zone
    type_ = 'a'
    description = 'Test recordset'
    ttl = 3600
    records = ['192.168.1.1']
    self.addDetail('zone', content.text_content(zone))
    self.addDetail('recordset', content.text_content(name))
    zone_obj = self.user_cloud.create_zone(name=zone, email=email)
    created_recordset = self.user_cloud.create_recordset(zone, name, type_, records, description, ttl)
    self.addCleanup(self.cleanup, zone, created_recordset['id'])
    self.assertEqual(created_recordset['zone_id'], zone_obj['id'])
    self.assertEqual(created_recordset['name'], name)
    self.assertEqual(created_recordset['type'], type_.upper())
    self.assertEqual(created_recordset['records'], records)
    self.assertEqual(created_recordset['description'], description)
    self.assertEqual(created_recordset['ttl'], ttl)
    recordsets = self.user_cloud.list_recordsets(zone)
    self.assertIsNotNone(recordsets)
    get_recordset = self.user_cloud.get_recordset(zone, created_recordset['id'])
    self.assertEqual(get_recordset['id'], created_recordset['id'])
    updated_recordset = self.user_cloud.update_recordset(zone_obj['id'], created_recordset['id'], ttl=7200)
    self.assertEqual(updated_recordset['id'], created_recordset['id'])
    self.assertEqual(updated_recordset['name'], name)
    self.assertEqual(updated_recordset['type'], type_.upper())
    self.assertEqual(updated_recordset['records'], records)
    self.assertEqual(updated_recordset['description'], description)
    self.assertEqual(updated_recordset['ttl'], 7200)
    deleted_recordset = self.user_cloud.delete_recordset(zone, created_recordset['id'])
    self.assertTrue(deleted_recordset)