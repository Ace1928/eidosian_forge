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
def krm_kind_to_apitools_collection_name(krm_kind, krm_group, apitools_collection_names):
    """Converts the config-connector resource name to apitools collection name.

  Applies several heuristics based on commonalities between KRM Group and Kind
  values and apitools collection names toto map the KRM Group and Kind to the
  apitools collection name.

  Args:
    krm_kind: The KRM Kind provided by the config connector binary.
    krm_group: The KRM group provided by the config-connector binary.
    apitools_collection_names: Set of all collections for the relevant service.

  Raises:
    KrmToApitoolsResourceNameError: Raised if no apitools collection name
      could be derived for the given krm_kind and krm_group.

  Returns:
    The converted resource name.
  """
    apitools_collection_guess = krm_kind
    apitools_collection_guess = remove_krm_group(apitools_collection_guess, krm_group)
    apitools_collection_guess = name_parsing.pluralize(apitools_collection_guess)
    apitools_collection_guess = lowercase_first_segment(apitools_collection_guess)
    apitools_collection_guess = capitalize_interior_acronyms(apitools_collection_guess)
    if apitools_collection_guess in apitools_collection_names:
        return apitools_collection_guess
    possible_matches = find_possible_matches(apitools_collection_guess, apitools_collection_names)
    best_match = pick_best_match(possible_matches)
    if best_match:
        return best_match
    else:
        raise KrmToApitoolsResourceNameError('Cant match: {}: {}'.format(krm_group, krm_kind))