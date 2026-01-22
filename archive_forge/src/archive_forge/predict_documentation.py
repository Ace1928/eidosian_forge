from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ml_engine import flags
from googlecloudsdk.command_lib.ml_engine import local_utils
from googlecloudsdk.command_lib.ml_engine import predict_utilities
from googlecloudsdk.core import log
Run prediction locally.