from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import collections
import six
from pasta.base import annotate
from pasta.base import formatting as fmt
from pasta.base import fstring_utils
def check_is_continued_try(self, node):
    return getattr(node, 'is_continued', False)