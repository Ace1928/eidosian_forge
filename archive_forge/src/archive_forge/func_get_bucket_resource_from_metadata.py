from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from apitools.base.py import encoding
from apitools.base.py import encoding_helper
from googlecloudsdk.api_lib.storage import metadata_util
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.api_lib.storage.gcs_json import metadata_field_converters
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import gzip_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.command_lib.storage.resources import gcs_resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core import properties
def get_bucket_resource_from_metadata(metadata):
    """Helper method to generate a BucketResource instance from GCS metadata.

  Args:
    metadata (messages.Bucket): Extract resource properties from this.

  Returns:
    BucketResource with properties populated by metadata.
  """
    url = storage_url.CloudUrl(scheme=storage_url.ProviderPrefix.GCS, bucket_name=metadata.name)
    if metadata.autoclass and metadata.autoclass.enabled:
        autoclass_enabled_time = metadata.autoclass.toggleTime
    else:
        autoclass_enabled_time = None
    uniform_bucket_level_access = getattr(getattr(metadata.iamConfiguration, 'uniformBucketLevelAccess', None), 'enabled', None)
    return gcs_resource_reference.GcsBucketResource(url, acl=_message_to_dict(metadata.acl), autoclass=_message_to_dict(metadata.autoclass), autoclass_enabled_time=autoclass_enabled_time, cors_config=_message_to_dict(metadata.cors), creation_time=metadata.timeCreated, custom_placement_config=_message_to_dict(metadata.customPlacementConfig), default_acl=_message_to_dict(metadata.defaultObjectAcl), default_event_based_hold=metadata.defaultEventBasedHold or None, default_kms_key=getattr(metadata.encryption, 'defaultKmsKeyName', None), default_storage_class=metadata.storageClass, etag=metadata.etag, labels=_message_to_dict(metadata.labels), lifecycle_config=_message_to_dict(metadata.lifecycle), location=metadata.location, location_type=metadata.locationType, logging_config=_message_to_dict(metadata.logging), metadata=metadata, metageneration=metadata.metageneration, per_object_retention=_message_to_dict(metadata.objectRetention), project_number=metadata.projectNumber, public_access_prevention=getattr(metadata.iamConfiguration, 'publicAccessPrevention', None), requester_pays=getattr(metadata.billing, 'requesterPays', None), retention_policy=_message_to_dict(metadata.retentionPolicy), rpo=metadata.rpo, satisfies_pzs=metadata.satisfiesPZS, soft_delete_policy=_message_to_dict(metadata.softDeletePolicy), uniform_bucket_level_access=uniform_bucket_level_access, update_time=metadata.updated, versioning_enabled=getattr(metadata.versioning, 'enabled', None), website_config=_message_to_dict(metadata.website))