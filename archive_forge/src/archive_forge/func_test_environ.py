from collections import defaultdict as _defaultdict
from collections.abc import Mapping
import os
from toolz.dicttoolz import (merge, merge_with, valmap, keymap, update_in,
from toolz.functoolz import identity
from toolz.utils import raises
def test_environ():
    assert keymap(identity, os.environ) == os.environ
    assert valmap(identity, os.environ) == os.environ
    assert itemmap(identity, os.environ) == os.environ