from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.storage import errors as cloud_errors
def get_status_code(error):
    if error.response:
        return error.response.get('status')