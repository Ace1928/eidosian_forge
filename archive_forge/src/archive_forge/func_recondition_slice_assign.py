from kivy.properties import ListProperty, ObservableDict, ObjectProperty
from kivy.event import EventDispatcher
from functools import partial
def recondition_slice_assign(val, last_len, new_len):
    if not isinstance(val, slice):
        return slice(val, val + 1)
    diff = new_len - last_len
    start, stop, step = (val.start, val.stop, val.step)
    if stop <= start:
        return slice(0, 0)
    if step is not None and step != 1:
        assert last_len == new_len
        if stop < 0:
            stop = max(0, last_len + stop)
        stop = min(last_len, stop)
        if start < 0:
            start = max(0, last_len + start)
        start = min(last_len, start)
        return slice(start, stop, step)
    if start < 0:
        start = last_len + start
    if stop < 0:
        stop = last_len + stop
    if start < 0 or stop < 0 or start > last_len or (stop > last_len) or (new_len != last_len):
        return None
    return slice(start, stop)