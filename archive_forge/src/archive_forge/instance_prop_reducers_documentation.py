from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import datetime
from googlecloudsdk.api_lib.sql import api_util as common_api_util
from googlecloudsdk.api_lib.sql import constants
from googlecloudsdk.api_lib.sql import exceptions as sql_exceptions
from googlecloudsdk.api_lib.sql import instances as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
Generates or clears password policy for the instance.

  Args:
    sql_messages: module, The messages module that should be used.
    password_policy_min_length: int, Minimum number of characters allowed.
    password_policy_complexity: string, The complexity of the password.
    password_policy_reuse_interval: int, Number of previous passwords that
      cannot be reused.
    password_policy_disallow_username_substring: boolean, True if disallow
      username as a part of the password.
    password_policy_password_change_interval: duration, Minimum interval at
      which password can be changed.
    enable_password_policy: boolean, True if password validation policy is
      enabled.
    clear_password_policy: boolean, True if clear existing password policy.

  Returns:
    sql_messages.PasswordValidationPolicy or None
  