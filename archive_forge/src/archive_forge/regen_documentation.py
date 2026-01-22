from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import fnmatch
import os
import re
import shutil
import googlecloudsdk
from googlecloudsdk import third_party
from googlecloudsdk.api_lib.regen import generate
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.command_lib.meta import regen as regen_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import ruamel.yaml
import six
from six.moves import map
Loads regen config from given filename.