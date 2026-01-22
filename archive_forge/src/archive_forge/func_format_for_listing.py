import re
def format_for_listing(pool_list, _):
    """Format existing fields for displaying them in the list response.

  The formatting logic is complicated enough to the point gcloud"s formatter
  is inconvenient to use.

  Args:
    pool_list: list of storage pools.
  Returns:
    the inputted pool list, with the added fields containing new formatting.
  """
    return list(map(_format_single, pool_list))