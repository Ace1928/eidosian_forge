from unittest import mock
from testtools import matchers
from oslo_service import periodic_task
from oslo_service.tests import base
@periodic_task.periodic_task
def tack(self, context):
    pass