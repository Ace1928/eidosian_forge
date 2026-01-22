from oslo_utils import uuidutils
from taskflow.engines.action_engine import executor
from taskflow.engines.worker_based import protocol as pr
from taskflow import exceptions as excp
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
from taskflow.types import failure
def request_to_dict(self, **kwargs):
    to_dict = dict(task_cls=self.task.name, task_name=self.task.name, task_version=self.task.version, action=self.task_action, arguments=self.task_args)
    to_dict.update(kwargs)
    return to_dict