import re
from .checks import check_data, check_msgdict, check_value
from .decode import decode_message
from .encode import encode_message
from .specs import REALTIME_TYPES, SPEC_BY_TYPE, make_msgdict
from .strings import msg2str, str2msg
def is_cc(self, control=None):
    """Return True if the message is of type 'control_change'.

        The optional control argument can be used to test for a specific
        control number, for example:

        if msg.is_cc(7):
            # Message is control change 7 (channel volume).
        """
    if self.type != 'control_change':
        return False
    elif control is None:
        return True
    else:
        return self.control == control