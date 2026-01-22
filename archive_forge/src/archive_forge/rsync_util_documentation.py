from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
Constructor.

    Args:
      src_url_str: (str or None) The source URL string, or None if diff_action
          is REMOVE.
      dst_url_str: (str) The destination URL string.
      src_posix_attrs: (posix_util.POSIXAttributes) The source POSIXAttributes.
      diff_action: (DiffAction) DiffAction to be applied.
      copy_size: (int or None) The amount of bytes to copy, or None if
          diff_action is REMOVE.
    