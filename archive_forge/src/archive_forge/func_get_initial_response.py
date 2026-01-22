from binascii import crc32
from struct import unpack
from botocore.exceptions import EventStreamError
def get_initial_response(self):
    try:
        initial_event = next(self._event_generator)
        event_type = initial_event.headers.get(':event-type')
        if event_type == 'initial-response':
            return initial_event
    except StopIteration:
        pass
    raise NoInitialResponseError()