from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.command_lib.util.declarative.clients import kcc_client
from googlecloudsdk.command_lib.util.resource_map import base
from googlecloudsdk.command_lib.util.resource_map import resource_map_update_util
from googlecloudsdk.command_lib.util.resource_map.declarative import declarative_map
from googlecloudsdk.core import name_parsing
def lowercase_first_segment(apitools_collection_guess):
    """First segment of collection should be lowercased, handle acronyms."""
    acronyms = ['HTTPS', 'HTTP', 'SSL', 'URL', 'VPN', 'TCP']
    found_acronym = False
    for acronym in acronyms:
        if apitools_collection_guess.startswith(acronym):
            apitools_collection_guess = apitools_collection_guess.replace(acronym, acronym.lower())
            found_acronym = True
    if not found_acronym:
        apitools_collection_guess = apitools_collection_guess[0].lower() + apitools_collection_guess[1:]
    return apitools_collection_guess