import gc
import weakref
import pytest
def test_trigger_cancel(kivy_clock, clock_counter):
    trigger = kivy_clock.create_trigger(clock_counter, 5.0)
    trigger()
    trigger.cancel()
    kivy_clock.tick()
    assert clock_counter.counter == 0