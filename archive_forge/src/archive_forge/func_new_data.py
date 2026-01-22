import time
from .exceptions import EOF, TIMEOUT
def new_data(self, data):
    spawn = self.spawn
    freshlen = len(data)
    spawn._before.write(data)
    if not self.searchwindowsize:
        if self.lookback:
            old_len = spawn._buffer.tell()
            spawn._buffer.write(data)
            spawn._buffer.seek(max(0, old_len - self.lookback))
            window = spawn._buffer.read()
        else:
            spawn._buffer.write(data)
            window = spawn.buffer
    elif len(data) >= self.searchwindowsize or not spawn._buffer.tell():
        window = data[-self.searchwindowsize:]
        spawn._buffer = spawn.buffer_type()
        spawn._buffer.write(window[-self.searchwindowsize:])
    else:
        spawn._buffer.write(data)
        new_len = spawn._buffer.tell()
        spawn._buffer.seek(max(0, new_len - self.searchwindowsize))
        window = spawn._buffer.read()
    return self.do_search(window, freshlen)