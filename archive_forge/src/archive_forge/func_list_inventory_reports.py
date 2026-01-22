from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.api_lib.storage.gcs_json import client as gcs_json_client
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import properties
def list_inventory_reports(self, source_bucket=None, destination=None, location=None, page_size=None):
    """Lists the report configs.

    Args:
      source_bucket (storage_url.CloudUrl): Source bucket for which reports will
        be generated.
      destination (storage_url.CloudUrl): The destination url where the
        generated reports will be stored.
      location (str): The location for which the report configs should be
        listed.
      page_size (int|None): Number of items per request to be returend.

    Returns:
      List of Report configs.
    """
    if location is not None:
        parent = _get_parent_string(properties.VALUES.core.project.Get(), location)
    else:
        parent = _get_parent_string_from_bucket(source_bucket.bucket_name if source_bucket is not None else destination.bucket_name)
    return list_pager.YieldFromList(self.client.projects_locations_reportConfigs, self.messages.StorageinsightsProjectsLocationsReportConfigsListRequest(parent=parent, filter=self._get_filters_for_list(source_bucket, destination)), batch_size=page_size if page_size is not None else PAGE_SIZE, batch_size_attribute='pageSize', field='reportConfigs')