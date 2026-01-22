import copy
import re
import numpy as np
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.lib import debug_data
def locate_tensor_element(formatted, indices):
    """Locate a tensor element in formatted text lines, given element indices.

  Given a RichTextLines object representing a tensor and indices of the sought
  element, return the row number at which the element is located (if exists).

  Args:
    formatted: A RichTextLines object containing formatted text lines
      representing the tensor.
    indices: Indices of the sought element, as a list of int or a list of list
      of int. The former case is for a single set of indices to look up,
      whereas the latter case is for looking up a batch of indices sets at once.
      In the latter case, the indices must be in ascending order, or a
      ValueError will be raised.

  Returns:
    1) A boolean indicating whether the element falls into an omitted line.
    2) Row index.
    3) Column start index, i.e., the first column in which the representation
       of the specified tensor starts, if it can be determined. If it cannot
       be determined (e.g., due to ellipsis), None.
    4) Column end index, i.e., the column right after the last column that
       represents the specified tensor. Iff it cannot be determined, None.

  For return values described above are based on a single set of indices to
    look up. In the case of batch mode (multiple sets of indices), the return
    values will be lists of the types described above.

  Raises:
    AttributeError: If:
      Input argument "formatted" does not have the required annotations.
    ValueError: If:
      1) Indices do not match the dimensions of the tensor, or
      2) Indices exceed sizes of the tensor, or
      3) Indices contain negative value(s).
      4) If in batch mode, and if not all sets of indices are in ascending
         order.
  """
    if isinstance(indices[0], list):
        indices_list = indices
        input_batch = True
    else:
        indices_list = [indices]
        input_batch = False
    if 'tensor_metadata' not in formatted.annotations:
        raise AttributeError('tensor_metadata is not available in annotations.')
    _validate_indices_list(indices_list, formatted)
    dims = formatted.annotations['tensor_metadata']['shape']
    batch_size = len(indices_list)
    lines = formatted.lines
    annot = formatted.annotations
    prev_r = 0
    prev_line = ''
    prev_indices = [0] * len(dims)
    are_omitted = [None] * batch_size
    row_indices = [None] * batch_size
    start_columns = [None] * batch_size
    end_columns = [None] * batch_size
    batch_pos = 0
    for r in range(len(lines)):
        if r not in annot:
            continue
        if BEGIN_INDICES_KEY in annot[r]:
            indices_key = BEGIN_INDICES_KEY
        elif OMITTED_INDICES_KEY in annot[r]:
            indices_key = OMITTED_INDICES_KEY
        matching_indices_list = [ind for ind in indices_list[batch_pos:] if prev_indices <= ind < annot[r][indices_key]]
        if matching_indices_list:
            num_matches = len(matching_indices_list)
            match_start_columns, match_end_columns = _locate_elements_in_line(prev_line, matching_indices_list, prev_indices)
            start_columns[batch_pos:batch_pos + num_matches] = match_start_columns
            end_columns[batch_pos:batch_pos + num_matches] = match_end_columns
            are_omitted[batch_pos:batch_pos + num_matches] = [OMITTED_INDICES_KEY in annot[prev_r]] * num_matches
            row_indices[batch_pos:batch_pos + num_matches] = [prev_r] * num_matches
            batch_pos += num_matches
            if batch_pos >= batch_size:
                break
        prev_r = r
        prev_line = lines[r]
        prev_indices = annot[r][indices_key]
    if batch_pos < batch_size:
        matching_indices_list = indices_list[batch_pos:]
        num_matches = len(matching_indices_list)
        match_start_columns, match_end_columns = _locate_elements_in_line(prev_line, matching_indices_list, prev_indices)
        start_columns[batch_pos:batch_pos + num_matches] = match_start_columns
        end_columns[batch_pos:batch_pos + num_matches] = match_end_columns
        are_omitted[batch_pos:batch_pos + num_matches] = [OMITTED_INDICES_KEY in annot[prev_r]] * num_matches
        row_indices[batch_pos:batch_pos + num_matches] = [prev_r] * num_matches
    if input_batch:
        return (are_omitted, row_indices, start_columns, end_columns)
    else:
        return (are_omitted[0], row_indices[0], start_columns[0], end_columns[0])