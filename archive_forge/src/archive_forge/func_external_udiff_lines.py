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
def external_udiff_lines(old, new, use_stringio=False):
    if use_stringio:
        output = BytesIO()
    else:
        output = tempfile.TemporaryFile()
    try:
        diff.external_diff('old', old, 'new', new, output, diff_opts=['-u'])
    except errors.NoDiff:
        raise tests.TestSkipped('external "diff" not present to test')
    output.seek(0, 0)
    lines = output.readlines()
    output.close()
    return lines