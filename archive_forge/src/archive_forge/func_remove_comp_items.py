import copy
import os
import re
import traceback
import numpy as np
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.platform import gfile
def remove_comp_items(self, context_word, comp_items):
    """Remove a list of completion items from a completion context.

    Args:
      context_word: A single completion word as a string. The removal will
        also apply to all other context words of the same context.
      comp_items: Completion items to remove.

    Raises:
      KeyError: if the context word has not been registered.
    """
    if context_word not in self._comp_dict:
        raise KeyError('Context word "%s" has not been registered' % context_word)
    for item in comp_items:
        self._comp_dict[context_word].remove(item)