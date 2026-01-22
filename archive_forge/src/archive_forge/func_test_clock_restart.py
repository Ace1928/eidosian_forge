import gc
import weakref
import pytest
def test_clock_restart(kivy_clock):
    kivy_clock.stop_clock()
    kivy_clock.start_clock()