from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudresourcemanager import organizations
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import resources
Get the Organization ID for the provided Organization argument.

  Numeric values will be returned, values like 'organizations/123456789' will
  return '123456789' and a value like 'example.com' will search for the
  organization ID associated with that domain.

  Args:
    org_argument: The value of the organization argument.

  Returns:
    A string containing the numeric organization ID, or None if the
    organization ID could not be determined.
  