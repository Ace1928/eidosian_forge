from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage.tasks import task
Initializes a task instance.

    Args:
      source_resource (resource_reference.FileObjectResource): The file to
        upload.
      destination_resource (resource_reference.ObjectResource|UnknownResource):
        Destination metadata for the upload.
      length (int): The size of source_resource.
    