import pytest
from kivy.compat import isclose
from kivy.input import MotionEvent
def test_to_absolute_pos(self):
    event = self.create_dummy_motion_event()
    for item in self.build_to_absolute_pos_data(320, 240, 20, 21):
        args = item[:-1]
        expected_x, expected_y = item[-1]
        x, y = event.to_absolute_pos(*args)
        message = 'For args {} expected ({}, {}), got ({}, {})'.format(args, expected_x, expected_y, x, y)
        correct = isclose(x, expected_x) and isclose(y, expected_y)
        assert correct, message