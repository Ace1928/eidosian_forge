import os
import re
from io import BytesIO, StringIO
import yaml
from .. import registry, tests, version_info_formats
from ..bzr.rio import read_stanzas
from ..version_info_formats.format_custom import (CustomVersionInfoBuilder,
from ..version_info_formats.format_python import PythonVersionInfoBuilder
from ..version_info_formats.format_rio import RioVersionInfoBuilder
from ..version_info_formats.format_yaml import YamlVersionInfoBuilder
from . import TestCaseWithTransport
def create_tree_with_dotted_revno(self):
    wt = self.make_branch_and_tree('branch')
    self.build_tree(['branch/a'])
    wt.add('a')
    wt.commit('a', rev_id=b'r1')
    other = wt.controldir.sprout('other').open_workingtree()
    self.build_tree(['other/b.a'])
    other.add(['b.a'])
    other.commit('b.a', rev_id=b'o2')
    os.chdir('branch')
    self.run_bzr('merge ../other')
    wt.commit('merge', rev_id=b'merge')
    wt.update(revision=b'o2')
    return wt