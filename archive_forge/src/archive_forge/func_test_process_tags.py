from unittest import mock
from oslo_config import cfg
from osprofiler.drivers import jaeger
from osprofiler import opts
from osprofiler.tests import test
from jaeger_client import Config
def test_process_tags(self):
    tags = self.driver.tracer.tags
    del tags['hostname']
    del tags['jaeger.version']
    del tags['ip']
    self.assertEqual({'k1': 'v1', 'k2': 'v2'}, tags)