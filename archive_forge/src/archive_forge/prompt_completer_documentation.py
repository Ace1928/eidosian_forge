from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from googlecloudsdk.core.console import console_attr
from six.moves import range  # pylint: disable=redefined-builtin
Displays the possible completions and redraws the prompt and response.

    Args:
      prefix: str, The current response.
      matches: [str], The list of strings that start with prefix.
    