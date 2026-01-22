import gc
import weakref
import pytest
def test_clock_ended_del_safe(kivy_clock, clock_counter):
    counter2 = ClockCounter()
    kivy_clock.schedule_lifecycle_aware_del_safe(clock_counter, counter2)
    kivy_clock.stop_clock()
    assert clock_counter.counter == 0
    assert counter2.counter == 1