from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.rds import RDSConnection
from boto.rds.dbsnapshot import DBSnapshot
from boto.rds import DBInstance
def test_create_dbinstance(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.create_dbsnapshot('mydbsnapshot', 'simcoprod01')
    self.assert_request_parameters({'Action': 'CreateDBSnapshot', 'DBSnapshotIdentifier': 'mydbsnapshot', 'DBInstanceIdentifier': 'simcoprod01'}, ignore_params_values=['Version'])
    self.assertIsInstance(response, DBSnapshot)
    self.assertEqual(response.id, 'mydbsnapshot')
    self.assertEqual(response.instance_id, 'simcoprod01')
    self.assertEqual(response.status, 'creating')