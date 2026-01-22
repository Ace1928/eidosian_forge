from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from absl.flags import _helpers
@classmethod
def from_flag(cls, flagname, flag_values, other_flag_values=None):
    """Creates a DuplicateFlagError by providing flag name and values.

    Args:
      flagname: str, the name of the flag being redefined.
      flag_values: FlagValues, the FlagValues instance containing the first
          definition of flagname.
      other_flag_values: FlagValues, if it is not None, it should be the
          FlagValues object where the second definition of flagname occurs.
          If it is None, we assume that we're being called when attempting
          to create the flag a second time, and we use the module calling
          this one as the source of the second definition.

    Returns:
      An instance of DuplicateFlagError.
    """
    first_module = flag_values.find_module_defining_flag(flagname, default='<unknown>')
    if other_flag_values is None:
        second_module = _helpers.get_calling_module()
    else:
        second_module = other_flag_values.find_module_defining_flag(flagname, default='<unknown>')
    flag_summary = flag_values[flagname].help
    msg = "The flag '%s' is defined twice. First from %s, Second from %s.  Description from first occurrence: %s" % (flagname, first_module, second_module, flag_summary)
    return cls(msg)