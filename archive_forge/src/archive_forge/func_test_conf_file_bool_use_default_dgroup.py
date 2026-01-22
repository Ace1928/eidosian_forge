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
def test_conf_file_bool_use_default_dgroup(self):
    self._do_default_dgroup_test_use(cfg.BoolOpt, 'yes', True)