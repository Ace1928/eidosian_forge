from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.protorpclite import protojson
from apitools.base.py import encoding
from apitools.base.py import extra_types
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import arg_parsers
create partner metadata from the given args.

  Args:
    partner_metadata: partner metadata dictionary.
    partner_metadata_from_file: partner metadata file content.

  Returns:
    python dict contains partner metadata from given args.
  