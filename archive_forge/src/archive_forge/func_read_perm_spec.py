import os
def read_perm_spec(spec):
    """
    Reads a spec like 'rw-r--r--' into a octal number suitable for
    chmod.  That is characters in groups of three -- first group is
    user, second for group, third for other (all other people).  The
    characters are r (read), w (write), and x (executable), though the
    executable can also be s (sticky).  Files in sticky directories
    get the directories permission setting.

    Examples::

      >>> print oct(read_perm_spec('rw-r--r--'))
      0o644
      >>> print oct(read_perm_spec('rw-rwsr--'))
      0o2664
      >>> print oct(read_perm_spec('r-xr--r--'))
      0o544
      >>> print oct(read_perm_spec('r--------'))
      0o400
    """
    total_mask = 0
    set_bits = (2048, 1024, 0)
    pieces = (spec[0:3], spec[3:6], spec[6:9])
    for i, (mode, set_bit) in enumerate(zip(pieces, set_bits)):
        mask = 0
        read, write, exe = list(mode)
        if read == 'r':
            mask = mask | 4
        elif read != '-':
            raise ValueError("Character %r unexpected (should be '-' or 'r')" % read)
        if write == 'w':
            mask = mask | 2
        elif write != '-':
            raise ValueError("Character %r unexpected (should be '-' or 'w')" % write)
        if exe == 'x':
            mask = mask | 1
        elif exe not in ('s', '-'):
            raise ValueError("Character %r unexpected (should be '-', 'x', or 's')" % exe)
        if exe == 's' and i == 2:
            raise ValueError("The 'other' executable setting cannot be suid/sgid ('s')")
        mask = mask << (2 - i) * 3
        if exe == 's':
            mask = mask | set_bit
        total_mask = total_mask | mask
    return total_mask