import gc
import weakref
import pytest
def test_exception_ignored(kivy_clock, clock_counter):

    def raise_exception(*args):
        raise ValueError('Stooooop')
    kivy_clock.schedule_once(raise_exception)
    kivy_clock.schedule_once(clock_counter)
    with pytest.raises(ValueError):
        kivy_clock.tick()
    assert clock_counter.counter == 0