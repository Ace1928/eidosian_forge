from collections import abc
import contextlib
import dataclasses
import difflib
import enum
import errno
import faulthandler
import getpass
import inspect
import io
import itertools
import json
import os
import random
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import textwrap
import typing
from typing import Any, AnyStr, BinaryIO, Callable, ContextManager, IO, Iterator, List, Mapping, MutableMapping, MutableSequence, NoReturn, Optional, Sequence, Text, TextIO, Tuple, Type, Union
import unittest
from unittest import mock  # pylint: disable=unused-import Allow absltest.mock.
from urllib import parse
from absl import app  # pylint: disable=g-import-not-at-top
from absl import flags
from absl import logging
from absl.testing import _pretty_print_reporter
from absl.testing import xml_reporter
def get_default_xml_output_filename():
    if os.environ.get('XML_OUTPUT_FILE'):
        return os.environ['XML_OUTPUT_FILE']
    elif os.environ.get('RUNNING_UNDER_TEST_DAEMON'):
        return os.path.join(os.path.dirname(TEST_TMPDIR.value), 'test_detail.xml')
    elif os.environ.get('TEST_XMLOUTPUTDIR'):
        return os.path.join(os.environ['TEST_XMLOUTPUTDIR'], os.path.splitext(os.path.basename(sys.argv[0]))[0] + '.xml')