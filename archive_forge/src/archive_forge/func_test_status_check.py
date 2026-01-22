import datetime
from oslo_utils import timeutils
import uuid
from heat.common import service_utils
from heat.db import models
from heat.tests import common
def test_status_check(self):
    service = models.Service()
    service.id = str(uuid.uuid4())
    service.engine_id = str(uuid.uuid4())
    service.binary = 'heat-engine'
    service.hostname = 'host.devstack.org'
    service.host = 'engine-1'
    service.report_interval = 60
    service.topic = 'engine'
    service.created_at = timeutils.utcnow()
    service.deleted_at = None
    service.updated_at = None
    service_dict = service_utils.format_service(service)
    self.assertEqual(service_dict['id'], service.id)
    self.assertEqual(service_dict['engine_id'], service.engine_id)
    self.assertEqual(service_dict['host'], service.host)
    self.assertEqual(service_dict['hostname'], service.hostname)
    self.assertEqual(service_dict['binary'], service.binary)
    self.assertEqual(service_dict['topic'], service.topic)
    self.assertEqual(service_dict['report_interval'], service.report_interval)
    self.assertEqual(service_dict['created_at'], service.created_at)
    self.assertEqual(service_dict['updated_at'], service.updated_at)
    self.assertEqual(service_dict['deleted_at'], service.deleted_at)
    self.assertEqual(service_dict['status'], 'up')
    service_dict = service_utils.format_service(service)
    self.assertEqual(service_dict['status'], 'up')
    service.created_at = timeutils.utcnow() - datetime.timedelta(0, 130)
    service_dict = service_utils.format_service(service)
    self.assertEqual(service_dict['status'], 'down')
    service.updated_at = timeutils.utcnow() - datetime.timedelta(0, 130)
    service_dict = service_utils.format_service(service)
    self.assertEqual(service_dict['status'], 'down')
    service.updated_at = timeutils.utcnow() - datetime.timedelta(0, 50)
    service_dict = service_utils.format_service(service)
    self.assertEqual(service_dict['status'], 'up')