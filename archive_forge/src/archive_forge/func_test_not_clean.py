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
def test_not_clean(self):
    wt = self.create_branch()
    self.build_tree(['branch/c'])
    val = self.regen(wt, 'clean: {clean}', check_for_clean=True)
    self.assertEqual(val, 'clean: 0')
    os.remove('branch/c')