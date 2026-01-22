import gc
import weakref
import pytest
def test_unschedule(kivy_clock, clock_counter):
    kivy_clock.schedule_once(clock_counter)
    kivy_clock.unschedule(clock_counter)
    kivy_clock.tick()
    assert clock_counter.counter == 0