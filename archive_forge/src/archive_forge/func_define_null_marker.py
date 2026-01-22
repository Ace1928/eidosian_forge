from typing import Optional
from absl import flags
def define_null_marker(flag_values: flags.FlagValues) -> flags.FlagHolder[Optional[str]]:
    return flags.DEFINE_string('null_marker', None, 'An optional custom string that will represent a NULL valuein CSV External table data.', flag_values=flag_values)