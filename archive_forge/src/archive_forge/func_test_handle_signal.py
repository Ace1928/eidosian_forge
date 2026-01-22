from unittest import mock
from heat.common import template_format
from heat.engine.clients.os import sahara
from heat.engine.resources.openstack.sahara import job
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_handle_signal(self):
    jb = self._create_resource('job', self.rsrc_defn, self.stack)
    scheduler.TaskRunner(jb.handle_signal, None)()
    expected_args = {'job_id': 'fake-resource-id', 'cluster_id': 'some res id', 'input_id': 'some res id', 'output_id': 'some res id', 'is_public': True, 'is_protected': False, 'interface': {}, 'configs': {'configs': {'mapred.reduce.class': 'org.apache.oozie.example.SampleReducer', 'mapred.map.class': 'org.apache.oozie.example.SampleMapper', 'mapreduce.framework.name': 'yarn'}, 'args': [], 'params': {}}}
    self.client.job_executions.create.assert_called_once_with(**expected_args)