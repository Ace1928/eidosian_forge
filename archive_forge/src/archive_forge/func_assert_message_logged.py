import argparse
import errno
import functools
import io
import logging
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock
import fixtures
from oslotest import base
import testscenarios
from oslo_config import cfg
from oslo_config import types
def assert_message_logged(self, deprecated_name, deprecated_group, current_name, current_group):
    expected = cfg._Namespace._deprecated_opt_message % {'dep_option': deprecated_name, 'dep_group': deprecated_group, 'option': current_name, 'group': current_group}
    self.assertIn(expected + '\n', self.log_fixture.output)