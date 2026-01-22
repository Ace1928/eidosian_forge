import contextlib
import os
import re
import subprocess
import sys
import tempfile
from io import BytesIO
from .. import diff, errors, osutils
from .. import revision as _mod_revision
from .. import revisionspec, revisiontree, tests
from ..tests import EncodingAdapter, features
from ..tests.scenarios import load_tests_apply_scenarios
def udiff_lines(old, new, allow_binary=False):
    output = BytesIO()
    diff.internal_diff('old', old, 'new', new, output, allow_binary)
    output.seek(0, 0)
    return output.readlines()