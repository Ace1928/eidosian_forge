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
def test_opt_not_eq(self):
    d1 = cfg.ListOpt('oldfoo')
    d2 = cfg.ListOpt('oldbar')
    self.assertNotEqual(d1, d2)