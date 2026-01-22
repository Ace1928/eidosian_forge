import gc
import weakref
import pytest
def test_exception_caught(kivy_clock, clock_counter):
    exception = None

    def handle_test_exception(e):
        nonlocal exception
        exception = str(e)
    kivy_clock.handle_exception = handle_test_exception

    def raise_exception(*args):
        raise ValueError('Stooooop')
    kivy_clock.schedule_once(raise_exception)
    kivy_clock.schedule_once(clock_counter)
    kivy_clock.tick()
    assert clock_counter.counter == 1
    assert exception == 'Stooooop'