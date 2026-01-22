from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.ec2.securitygroup import SecurityGroup
from boto.rds import RDSConnection
from boto.rds.vpcsecuritygroupmembership import VPCSecurityGroupMembership
from boto.rds.parametergroup import ParameterGroup
from boto.rds.logfile import LogFile, LogFileObject
import xml.sax.saxutils as saxutils
def test_describe_option_groups(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.describe_option_groups()
    self.assertEqual(len(response), 2)
    options = response[0]
    self.assertEqual(options.name, 'myoptiongroup')
    self.assertEqual(options.description, 'Test option group')
    self.assertEqual(options.engine_name, 'oracle-se1')
    self.assertEqual(options.major_engine_version, '11.2')
    options = response[1]
    self.assertEqual(options.name, 'default:oracle-se1-11-2')
    self.assertEqual(options.description, 'Default Option Group.')
    self.assertEqual(options.engine_name, 'oracle-se1')
    self.assertEqual(options.major_engine_version, '11.2')