import os
import ovs.util
import ovs.vlog
def get_max_tries(self):
    """Returns the current remaining number of connection attempts,
        None if the number is unlimited."""
    return self.max_tries