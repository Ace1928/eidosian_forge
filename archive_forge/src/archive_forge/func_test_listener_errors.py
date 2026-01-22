import os
import sys
import threading
import time
import unittest.mock
import pytest
from cherrypy.process import wspbus
def test_listener_errors(bus, listener):
    """Test that unhandled exceptions raise channel failures."""
    expected = []
    channels = [c for c in bus.listeners if c != 'log']
    for channel in channels:
        bus.subscribe(channel, listener.get_listener(channel, 1))
        bus.subscribe(channel, lambda: None, priority=20)
    for channel in channels:
        with pytest.raises(wspbus.ChannelFailures):
            bus.publish(channel, 123)
        expected.append(msg % (1, channel, 123))
    assert listener.responses == expected