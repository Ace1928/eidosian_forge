from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_tsigkey_name
from designateclient.functionaltests.datagen import random_tsigkey_secret
from designateclient.functionaltests.datagen import random_zone_name
from designateclient.functionaltests.v2.fixtures import TSIGKeyFixture
from designateclient.functionaltests.v2.fixtures import ZoneFixture
def test_tsigkey_create_and_show(self):
    tsigkey = self.clients.as_user('admin').tsigkey_show(self.tsigkey.id)
    self.assertEqual(tsigkey.name, self.tsigkey.name)
    self.assertEqual(tsigkey.created_at, self.tsigkey.created_at)
    self.assertEqual(tsigkey.id, self.tsigkey.id)
    self.assertEqual(self.tsig.algorithm, self.tsig_algorithm)
    self.assertEqual(self.tsig.secret, self.tsig_secret)
    self.assertEqual(self.tsig.scope, self.tsig_scope)
    self.assertEqual(self.tsig.resource_id, self.zone.id)
    self.assertEqual(tsigkey.updated_at, self.tsigkey.updated_at)