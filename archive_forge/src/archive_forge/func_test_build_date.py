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
def test_build_date(self):
    wt = self.create_branch()
    val = self.regen(wt, 'build-date: "{build_date}"\ndate: "{date}"')
    self.assertContainsRe(val, 'build-date: "[0-9-+: ]+"')
    self.assertContainsRe(val, 'date: "[0-9-+: ]+"')