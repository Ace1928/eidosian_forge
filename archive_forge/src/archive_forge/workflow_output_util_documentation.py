from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import custom_printer_base
Transform ApplicationStatus into the output structure of marker classes.

    Args:
      record: a dict object

    Returns:
      lines formatted for output
    