import gc
import weakref
import pytest
def test_unschedule_after_tick(kivy_clock, clock_counter):
    kivy_clock.schedule_once(clock_counter, 5.0)
    kivy_clock.tick()
    kivy_clock.unschedule(clock_counter)
    kivy_clock.tick()
    assert clock_counter.counter == 0