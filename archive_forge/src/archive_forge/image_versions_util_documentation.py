from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.api_lib.composer import image_versions_util as image_version_api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import flags
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core.util import semver
Validates whether version candidate is greater than or equal to current.

  Applicable both for Airflow and Composer version upgrades. Composer supports
  both Airflow and self MINOR and PATCH-level upgrades.

  Args:
    cur_version: current 'a.b.c' version
    candidate_version: candidate 'x.y.z' version
    image_version_part: part of image to be validated. Must be either 'Airflow'
      or 'Composer'

  Returns:
    UpgradeValidator namedtuple containing boolean value whether selected image
    version component is valid for upgrade and eventual error message if it is
    not.
  