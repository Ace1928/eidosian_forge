from __future__ import (absolute_import, division, print_function)
import importlib
import os
import re
import sys
import textwrap
import yaml
def load_doc_fragment(name):
    fn = doc_fragment_fn(name)
    return DocFragmentFile(fn)