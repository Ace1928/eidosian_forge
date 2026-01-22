from __future__ import absolute_import
import codecs
import os
import sys
import threading
import serial
from serial.tools.list_ports import comports
from serial.tools import hexlify_codec
def suspend_port(self):
    """        open port temporarily, allow reconnect, exit and port change to get
        out of the loop
        """
    self._stop_reader()
    self.serial.close()
    sys.stderr.write('\n--- Port closed: {} ---\n'.format(self.serial.port))
    do_change_port = False
    while not self.serial.is_open:
        sys.stderr.write('--- Quit: {exit} | p: port change | any other key to reconnect ---\n'.format(exit=key_description(self.exit_character)))
        k = self.console.getkey()
        if k == self.exit_character:
            self.stop()
            break
        elif k in 'pP':
            do_change_port = True
            break
        try:
            self.serial.open()
        except Exception as e:
            sys.stderr.write('--- ERROR opening port: {} ---\n'.format(e))
    if do_change_port:
        self.change_port()
    else:
        self._start_reader()
        sys.stderr.write('--- Port opened: {} ---\n'.format(self.serial.port))