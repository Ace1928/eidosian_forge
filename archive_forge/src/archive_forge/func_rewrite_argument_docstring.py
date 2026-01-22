import collections
import functools
import inspect
import re
from tensorflow.python.framework import strict_mode
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import decorator_utils
from tensorflow.python.util import is_in_graph_mode
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.tools.docs import doc_controls
def rewrite_argument_docstring(old_doc, old_argument, new_argument):
    return old_doc.replace('`%s`' % old_argument, '`%s`' % new_argument).replace('%s:' % old_argument, '%s:' % new_argument)