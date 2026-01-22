import gc
import weakref
import pytest
def test_clock_ended_raises(kivy_clock, clock_counter):
    from kivy.clock import ClockNotRunningError
    event = kivy_clock.create_lifecycle_aware_trigger(clock_counter, clock_counter)
    kivy_clock.stop_clock()
    with pytest.raises(ClockNotRunningError):
        event()
    assert clock_counter.counter == 0
    event = kivy_clock.create_lifecycle_aware_trigger(clock_counter, clock_counter)
    with pytest.raises(ClockNotRunningError):
        event()
    assert clock_counter.counter == 0
    kivy_clock.schedule_once(clock_counter)
    assert clock_counter.counter == 0