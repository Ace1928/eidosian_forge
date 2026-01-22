from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def get_update_mask(args, args_to_update_masks) -> str:
    """Maps user provided arguments to API supported mutable fields in format of yaml field paths.

  Args:
    args: All arguments passed from CLI.
    args_to_update_masks: Mapping for a specific resource, such as user cluster,
      or node pool.

  Returns:
    A string that contains yaml field paths to be used in the API update
    request.
  """
    update_mask_list = []
    for arg in args_to_update_masks:
        if hasattr(args, arg) and args.IsSpecified(arg):
            update_mask_list.append(args_to_update_masks[arg])
    return ','.join(sorted(set(update_mask_list)))