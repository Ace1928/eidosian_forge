from urllib import parse
from zunclient import api_versions
from zunclient.common import base
from zunclient.common import utils
from zunclient import exceptions
def execute_resize(self, id, exec_id, width, height):
    self._action(id, '/execute_resize', qparams={'exec_id': exec_id, 'w': width, 'h': height})[1]