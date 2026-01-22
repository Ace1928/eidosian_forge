from __future__ import absolute_import
import cython
import hashlib
import operator
import os
import re
import shutil
import textwrap
from string import Template
from functools import partial
from contextlib import closing, contextmanager
from collections import defaultdict
from . import Naming
from . import Options
from . import DebugFlags
from . import StringEncoding
from .. import Utils
from .Scanning import SourceDescriptor
from ..StringIOTree import StringIOTree
def label_interceptor(self, new_labels, orig_labels, skip_to_label=None, pos=None, trace=True):
    """
        Helper for generating multiple label interceptor code blocks.

        @param new_labels: the new labels that should be intercepted
        @param orig_labels: the original labels that we should dispatch to after the interception
        @param skip_to_label: a label to skip to before starting the code blocks
        @param pos: the node position to mark for each interceptor block
        @param trace: add a trace line for the pos marker or not
        """
    for label, orig_label in zip(new_labels, orig_labels):
        if not self.label_used(label):
            continue
        if skip_to_label:
            self.put_goto(skip_to_label)
            skip_to_label = None
        if pos is not None:
            self.mark_pos(pos, trace=trace)
        self.put_label(label)
        yield (label, orig_label)
        self.put_goto(orig_label)