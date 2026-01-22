import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
def py3_info(info):
    info_py3 = copy.deepcopy(info)
    new_name = re.sub('FakeTrace[^.]*', 'FakeTracedCls', info_py3['function']['name'])
    info_py3['function']['name'] = new_name
    return info_py3