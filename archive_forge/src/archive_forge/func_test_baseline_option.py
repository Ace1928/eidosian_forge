import gzip
import os
import re
import tempfile
from .... import tests
from ....tests import features
from ....tests.blackbox import ExternalBase
from ..cmds import _get_source_stream
from . import FastimportFeature
from :1
from :2
from :1
from :2
def test_baseline_option(self):
    tree = self.make_branch_and_tree('bl')
    with open('bl/a', 'w') as f:
        f.write('test 1')
    tree.add('a')
    tree.commit(message='add a')
    with open('bl/b', 'w') as f:
        f.write('test 2')
    with open('bl/a', 'a') as f:
        f.write('\ntest 3')
    tree.add('b')
    tree.commit(message='add b, modify a')
    with open('bl/c', 'w') as f:
        f.write('test 4')
    tree.add('c')
    tree.remove('b')
    tree.commit(message='add c, remove b')
    with open('bl/a', 'a') as f:
        f.write('\ntest 5')
    tree.commit(message='modify a again')
    with open('bl/d', 'w') as f:
        f.write('test 6')
    tree.add('d')
    tree.commit(message='add d')
    data = self.run_bzr('fast-export --baseline -r 3.. bl')[0]
    data = re.sub('committer.*', 'committer', data)
    self.assertIn(data, (fast_export_baseline_data1, fast_export_baseline_data2))
    data1 = self.run_bzr('fast-export --baseline bl')[0]
    data2 = self.run_bzr('fast-export bl')[0]
    self.assertEqual(data1, data2)