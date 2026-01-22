import boto.swf.layer2
from boto.swf.layer2 import Domain, ActivityType, WorkflowType, WorkflowExecution
from tests.unit import unittest
from mock import Mock
def test_domain_instantiation(self):
    self.assertEquals('test-domain', self.domain.name)
    self.assertEquals('My test domain', self.domain.description)