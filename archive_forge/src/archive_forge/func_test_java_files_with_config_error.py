import logging
import mock
import os
import re
import sys
import shutil
import tempfile
import textwrap
import unittest
from gae_ext_runtime import testutil
from gae_ext_runtime import ext_runtime
import constants
def test_java_files_with_config_error(self):
    self.write_file('foo.war', '')
    errors = []

    def ErrorFake(message):
        errors.append(message)
    config = testutil.AppInfoFake(runtime='java', env='2', runtime_config=dict(jdk='openjdk9'))
    with mock.patch.dict(ext_runtime._LOG_FUNCS, {'error': ErrorFake}):
        self.assertFalse(self.generate_configs(appinfo=config, deploy=True))
    self.assertEqual(errors, ['Unknown JDK : openjdk9.'])