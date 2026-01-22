import unittest
import pygame
def test_midis2events__extra_event_data_missing_timestamp(self):
    """Ensures midi events with extra data and no timestamps are handled
        properly.
        """
    midi_event_extra_data_no_timestamp = ((192, 0, 1, 2, 'extra'),)
    with self.assertRaises(ValueError):
        events = pygame.midi.midis2events([midi_event_extra_data_no_timestamp], 0)