from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console.style import ansi
from googlecloudsdk.core.console.style import text
Creates a StyleMapping object to be used by a StyledLogger.

    Args:
      mappings: (dict[TextTypes, TextAttributes]), the mapping
        to be used for this StyleMapping object.
    