from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import enum
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import folder_util
from googlecloudsdk.command_lib.storage import plurality_checkable_iterator
from googlecloudsdk.command_lib.storage import wildcard_iterator
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import shim_format_util
import six
def list_urls(self):
    all_sources_total_bytes = 0
    for url in self._cloud_urls:
        if self._total:
            all_sources_total_bytes += self._list_url(url)
        else:
            self._list_url(url)
    if self._total:
        self._print_total(all_sources_total_bytes)