from unittest import mock
from testtools import matchers
from oslo_service import periodic_task
from oslo_service.tests import base
@mock.patch('oslo_service.periodic_task.now')
def test_called_thrice(self, mock_now):
    time = 340
    mock_now.return_value = time

    class AService(periodic_task.PeriodicTasks):

        def __init__(self, conf):
            super(AService, self).__init__(conf)
            self.called = {'doit': 0, 'urg': 0, 'ticks': 0, 'tocks': 0}

        @periodic_task.periodic_task
        def doit(self, context):
            self.called['doit'] += 1

        @periodic_task.periodic_task
        def crashit(self, context):
            self.called['urg'] += 1
            raise AnException('urg')

        @periodic_task.periodic_task(spacing=10 + periodic_task.DEFAULT_INTERVAL, run_immediately=True)
        def doit_with_ticks(self, context):
            self.called['ticks'] += 1

        @periodic_task.periodic_task(spacing=10 + periodic_task.DEFAULT_INTERVAL)
        def doit_with_tocks(self, context):
            self.called['tocks'] += 1
    external_called = {'ext1': 0, 'ext2': 0}

    @periodic_task.periodic_task
    def ext1(self, context):
        external_called['ext1'] += 1

    @periodic_task.periodic_task(spacing=10 + periodic_task.DEFAULT_INTERVAL)
    def ext2(self, context):
        external_called['ext2'] += 1
    serv = AService(self.conf)
    serv.add_periodic_task(ext1)
    serv.add_periodic_task(ext2)
    serv.run_periodic_tasks(None)
    self.assertEqual(0, serv.called['doit'])
    self.assertEqual(0, serv.called['urg'])
    self.assertEqual(1, serv.called['ticks'])
    self.assertEqual(0, serv.called['tocks'])
    self.assertEqual(0, external_called['ext1'])
    self.assertEqual(0, external_called['ext2'])
    time = time + periodic_task.DEFAULT_INTERVAL
    mock_now.return_value = time
    serv.run_periodic_tasks(None)
    self.assertEqual(1, serv.called['doit'])
    self.assertEqual(1, serv.called['urg'])
    self.assertEqual(1, serv.called['ticks'])
    self.assertEqual(0, serv.called['tocks'])
    self.assertEqual(1, external_called['ext1'])
    self.assertEqual(0, external_called['ext2'])
    time = time + periodic_task.DEFAULT_INTERVAL / 2
    mock_now.return_value = time
    serv.run_periodic_tasks(None)
    self.assertEqual(1, serv.called['doit'])
    self.assertEqual(1, serv.called['urg'])
    self.assertEqual(2, serv.called['ticks'])
    self.assertEqual(1, serv.called['tocks'])
    self.assertEqual(1, external_called['ext1'])
    self.assertEqual(1, external_called['ext2'])
    time = time + periodic_task.DEFAULT_INTERVAL
    mock_now.return_value = time
    serv.run_periodic_tasks(None)
    self.assertEqual(2, serv.called['doit'])
    self.assertEqual(2, serv.called['urg'])
    self.assertEqual(3, serv.called['ticks'])
    self.assertEqual(2, serv.called['tocks'])
    self.assertEqual(2, external_called['ext1'])
    self.assertEqual(2, external_called['ext2'])