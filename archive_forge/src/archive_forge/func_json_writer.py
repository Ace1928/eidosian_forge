from __future__ import (absolute_import, division, print_function)
import csv
import datetime
import os
import time
import threading
from abc import ABCMeta, abstractmethod
from functools import partial
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.six import with_metaclass
from ansible.parsing.ajson import AnsibleJSONEncoder, json
from ansible.plugins.callback import CallbackBase
def json_writer(writer, timestamp, task_name, task_uuid, value):
    data = {'timestamp': timestamp, 'task_name': task_name, 'task_uuid': task_uuid, 'value': value}
    writer.write('%s%s%s' % (RS, json.dumps(data, cls=AnsibleJSONEncoder), LF))