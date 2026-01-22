import gc
import weakref
import pytest
def test_clock_ended_del_safe_raises(kivy_clock, clock_counter):
    from kivy.clock import ClockNotRunningError
    counter2 = ClockCounter()
    kivy_clock.stop_clock()
    with pytest.raises(ClockNotRunningError):
        kivy_clock.schedule_lifecycle_aware_del_safe(clock_counter, counter2)
    assert clock_counter.counter == 0