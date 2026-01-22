import os
def calc_set_mode(cur_mode, mode, keep_exe=True):
    """
    Calculates the new mode given the current node ``cur_mode`` and
    the mode spec ``mode`` and if ``keep_exe`` is true then also keep
    the executable bits in ``cur_mode`` if ``mode`` has no executable
    bits in it.  Return the new mode.

    Examples::

      >>> print oct(calc_set_mode(0o775, 0o644))
      0o755
      >>> print oct(calc_set_mode(0o775, 0o744))
      0o744
      >>> print oct(calc_set_mode(0o10600, 0o644))
      0o10644
      >>> print oct(calc_set_mode(0o775, 0o644, False))
      0o644
    """
    for exe_bit in exe_bits:
        if mode & exe_bit:
            keep_exe = False
    keep_parts = (cur_mode | full_mask) ^ full_mask
    if keep_exe:
        keep_parts = keep_parts | cur_mode & exe_mask
    new_mode = keep_parts | mode
    return new_mode