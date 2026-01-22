from __future__ import absolute_import
import io
import time
@rs485_mode.setter
def rs485_mode(self, rs485_settings):
    self._rs485_mode = rs485_settings
    if self.is_open:
        self._reconfigure_port()