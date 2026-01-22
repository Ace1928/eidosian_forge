from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.dataproc import util
def getTruncatedFieldNameBySuffix(self, suffix):
    """Get a field name by suffix and truncate it.

    The one_of fields in server response have their type name as field key.
    One can retrieve the name of those fields by iterating through all the
    fields.

    Args:
      suffix: the suffix to match.

    Returns:
      The first matched truncated field name.

    Raises:
      AttributeError: Error occur when there is no match for the suffix.

    Usage Example:
      In server response:
      {
        ...
        "sparkJob":{
          ...
        }
        ...
      }
      type = helper.getTruncatedFieldNameBySuffix('Job')
    """
    for field in [field.name for field in self._job.all_fields()]:
        if field.endswith(suffix):
            token, _, _ = field.rpartition(suffix)
            if self._job.get_assigned_value(field):
                return token
    raise AttributeError('Response has no field with {} as suffix.'.format(suffix))