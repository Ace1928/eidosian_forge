import logging
import os
from oslo_config import cfg
from osprofiler.drivers import base
from osprofiler import initializer
from osprofiler import opts
from osprofiler import profiler
from osprofiler.tests import test
def test_list_traces(self):
    initializer.init_from_conf(CONF, {}, self.PROJECT, self.SERVICE, 'host')
    profiler.init('SECRET_KEY')
    base_id = profiler.get().get_base_id()
    foo = Foo()
    foo.bar(1)
    engine = base.get_driver(CONF.profiler.connection_string, project=self.PROJECT, service=self.SERVICE, host='host', conf=CONF)
    traces = engine.list_traces()
    LOG.debug('Collected traces: %s', traces)
    self.assertIn(base_id, [t['base_id'] for t in traces])