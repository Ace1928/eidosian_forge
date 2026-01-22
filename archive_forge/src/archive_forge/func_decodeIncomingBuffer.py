import io
import logging
import os
import sys
import threading
import time
from io import StringIO
def decodeIncomingBuffer(self):
    if not self.encoding:
        self.output_buffer, self.decoder_buffer = (self.decoder_buffer, b'')
        return
    raw_len = len(self.decoder_buffer)
    chars = ''
    while raw_len:
        try:
            chars = self.decoder_buffer[:raw_len].decode(self.encoding)
            break
        except:
            pass
        raw_len -= 1
    if self.newlines is None:
        chars = chars.replace('\r\n', '\n').replace('\r', '\n')
    self.output_buffer += chars
    self.decoder_buffer = self.decoder_buffer[raw_len:]