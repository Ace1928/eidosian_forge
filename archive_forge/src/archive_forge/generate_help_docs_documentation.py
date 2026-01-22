from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import walker_util
from googlecloudsdk.command_lib.meta import help_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_attr
Runs generator and optionally updates help docs in directory.