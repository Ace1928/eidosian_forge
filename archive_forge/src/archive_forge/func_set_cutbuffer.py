from kivy.core.clipboard import ClipboardBase
def set_cutbuffer(self, data):
    if not isinstance(data, bytes):
        data = data.encode('utf8')
    p = self._clip('in', 'primary')
    p.communicate(data)