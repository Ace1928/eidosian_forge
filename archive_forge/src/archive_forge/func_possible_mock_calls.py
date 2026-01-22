import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
def possible_mock_calls(name, info):
    return [mock.call(name, info=info), mock.call(name, info=py3_info(info))]