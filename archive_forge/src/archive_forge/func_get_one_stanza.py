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
def get_one_stanza(self, stanza, key):
    new_stanzas = list(read_stanzas(BytesIO(stanza[key].encode('utf8'))))
    self.assertEqual(1, len(new_stanzas))
    return new_stanzas[0]