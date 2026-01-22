from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import os
import stat
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core import log
from googlecloudsdk.core.cache import function_result_cache
from googlecloudsdk.core.util import platforms
def get_posix_attributes_from_resource(resource, preserve_symlinks=False):
    """Parses unknown resource type for POSIX data."""
    if isinstance(resource, resource_reference.ObjectResource):
        return get_posix_attributes_from_cloud_resource(resource)
    if isinstance(resource, resource_reference.FileObjectResource):
        return get_posix_attributes_from_file(resource.storage_url.object_name, preserve_symlinks)
    raise errors.InvalidUrlError('Can only retrieve POSIX attributes from file or cloud object, not: {}'.format(resource.TYPE_STRING))