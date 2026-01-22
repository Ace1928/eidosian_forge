from midi devices.  It can also list midi devices on the system.
import math
import atexit
import pygame
import pygame.locals
import pygame.pypm as _pypm
def midis2events(midis, device_id):
    """converts midi events to pygame events
    pygame.midi.midis2events(midis, device_id): return [Event, ...]

    Takes a sequence of midi events and returns list of pygame events.
    """
    evs = []
    for midi in midis:
        (status, data1, data2, data3), timestamp = midi
        event = pygame.event.Event(MIDIIN, status=status, data1=data1, data2=data2, data3=data3, timestamp=timestamp, vice_id=device_id)
        evs.append(event)
    return evs