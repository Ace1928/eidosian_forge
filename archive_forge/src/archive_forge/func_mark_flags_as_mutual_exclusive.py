from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
from absl.flags import _exceptions
from absl.flags import _flagvalues
from absl.flags import _validators_classes
def mark_flags_as_mutual_exclusive(flag_names, required=False, flag_values=_flagvalues.FLAGS):
    """Ensures that only one flag among flag_names is not None.

  Important note: This validator checks if flag values are None, and it does not
  distinguish between default and explicit values. Therefore, this validator
  does not make sense when applied to flags with default values other than None,
  including other false values (e.g. False, 0, '', []). That includes multi
  flags with a default value of [] instead of None.

  Args:
    flag_names: [str], names of the flags.
    required: bool. If true, exactly one of the flags must have a value other
        than None. Otherwise, at most one of the flags can have a value other
        than None, and it is valid for all of the flags to be None.
    flag_values: flags.FlagValues, optional FlagValues instance where the flags
        are defined.
  """
    for flag_name in flag_names:
        if flag_values[flag_name].default is not None:
            warnings.warn('Flag --{} has a non-None default value. That does not make sense with mark_flags_as_mutual_exclusive, which checks whether the listed flags have a value other than None.'.format(flag_name), stacklevel=2)

    def validate_mutual_exclusion(flags_dict):
        flag_count = sum((1 for val in flags_dict.values() if val is not None))
        if flag_count == 1 or (not required and flag_count == 0):
            return True
        raise _exceptions.ValidationError('{} one of ({}) must have a value other than None.'.format('Exactly' if required else 'At most', ', '.join(flag_names)))
    register_multi_flags_validator(flag_names, validate_mutual_exclusion, flag_values=flag_values)