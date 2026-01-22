from warnings import warn
from .low_level import MessageType, HeaderFields
from .wrappers import DBusErrorResponse
def outgoing(self, msg):
    """Set the serial number in the message & make a handle if a method call
        """
    self.outgoing_serial += 1
    msg.header.serial = self.outgoing_serial
    if msg.header.message_type is MessageType.method_call:
        self.awaiting_reply[msg.header.serial] = handle = self.handle_factory()
        return handle