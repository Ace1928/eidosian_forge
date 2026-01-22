from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import errno
import os
import pdb
import sys
import textwrap
import traceback
from absl import command_name
from absl import flags
from absl import logging
def call_after_init(callback):
    """Calls the given callback only once ABSL has finished initialization.

  If ABSL has already finished initialization when `call_after_init` is
  called then the callback is executed immediately, otherwise `callback` is
  stored to be executed after `app.run` has finished initializing (aka. just
  before the main function is called).

  If called after `app.run`, this is equivalent to calling `callback()` in the
  caller thread. If called before `app.run`, callbacks are run sequentially (in
  an undefined order) in the same thread as `app.run`.

  Args:
    callback: a callable to be called once ABSL has finished initialization.
      This may be immediate if initialization has already finished. It
      takes no arguments and returns nothing.
  """
    if _run_init.done:
        callback()
    else:
        _init_callbacks.append(callback)