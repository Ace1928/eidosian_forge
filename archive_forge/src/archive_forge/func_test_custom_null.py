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
def test_custom_null(self):
    sio = StringIO()
    wt = self.make_branch_and_tree('branch')
    builder = CustomVersionInfoBuilder(wt.branch, working_tree=wt, template='revno: {revno}')
    builder.generate(sio)
    self.assertEqual('revno: 0', sio.getvalue())
    builder = CustomVersionInfoBuilder(wt.branch, working_tree=wt, template='{revno} revid: {revision_id}')
    self.assertRaises(MissingTemplateVariable, builder.generate, sio)