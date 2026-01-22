from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.rds import RDSConnection
from boto.rds.dbsnapshot import DBSnapshot
from boto.rds import DBInstance
def test_restore_dbinstance_from_dbsnapshot(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.restore_dbinstance_from_dbsnapshot('mydbsnapshot', 'myrestoreddbinstance', 'db.m1.large', '3306', 'us-east-1a', 'false', 'true')
    self.assert_request_parameters({'Action': 'RestoreDBInstanceFromDBSnapshot', 'DBSnapshotIdentifier': 'mydbsnapshot', 'DBInstanceIdentifier': 'myrestoreddbinstance', 'DBInstanceClass': 'db.m1.large', 'Port': '3306', 'AvailabilityZone': 'us-east-1a', 'MultiAZ': 'false', 'AutoMinorVersionUpgrade': 'true'}, ignore_params_values=['Version'])
    self.assertIsInstance(response, DBInstance)
    self.assertEqual(response.id, 'myrestoreddbinstance')
    self.assertEqual(response.status, 'creating')
    self.assertEqual(response.instance_class, 'db.m1.large')
    self.assertEqual(response.multi_az, False)