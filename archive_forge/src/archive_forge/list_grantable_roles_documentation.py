from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.iam import exceptions
from googlecloudsdk.api_lib.iam import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.iam import flags
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import resources
List IAM grantable roles for a resource.

  This command displays the list of grantable roles for a resource.
  The resource can be referenced either via the full resource name or via a URI.
  User can then add IAM policy bindings to grant the roles.
  