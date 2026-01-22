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
def get_diff_as_string(tree1, tree2, specific_files=None, working_tree=None):
    output = BytesIO()
    if working_tree is not None:
        extra_trees = (working_tree,)
    else:
        extra_trees = ()
    diff.show_diff_trees(tree1, tree2, output, specific_files=specific_files, extra_trees=extra_trees, old_label='old/', new_label='new/')
    return output.getvalue()