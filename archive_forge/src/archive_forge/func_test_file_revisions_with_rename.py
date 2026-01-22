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
def test_file_revisions_with_rename(self):
    wt = self.create_branch()
    self.build_tree(['branch/a', 'branch/c'])
    wt.add('c')
    wt.rename_one('b', 'd')
    stanza = self.regen(wt, check_for_clean=True, include_file_revisions=True)
    file_rev_stanza = stanza['file-revisions']
    self.assertEqual(['', 'a', 'b', 'c', 'd'], [r['path'] for r in file_rev_stanza])
    self.assertEqual(['r1', 'modified', 'renamed to d', 'new', 'renamed from b'], [r['revision'] for r in file_rev_stanza])