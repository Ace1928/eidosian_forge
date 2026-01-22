from __future__ import annotations
import abc
import time
import typing
from .common import BaseScreen
def get_input_nonblocking(self) -> tuple[None, list[str], list[int]]:
    """
        Return a (next_input_timeout, keys_pressed, raw_keycodes)
        tuple.

        The protocol for our device requires waiting for acks between
        each command, so this method responds to those as well as key
        press and release events.

        Key repeat events are simulated here as the device doesn't send
        any for us.

        raw_keycodes are the bytes of messages we received, which might
        not seem to have any correspondence to keys_pressed.
        """
    data_input: list[str] = []
    raw_data_input: list[int] = []
    timeout = None
    packet = self._read_packet()
    while packet:
        command, data = packet
        if command == self.CMD_KEY_ACTIVITY and data:
            d0 = data[0]
            if 1 <= d0 <= 12:
                release = d0 > 6
                keycode = d0 - release * 6 - 1
                key = self.key_map[keycode]
                if release:
                    self.key_repeat.release(key)
                else:
                    data_input.append(key)
                    self.key_repeat.press(key)
                raw_data_input.append(d0)
        elif command & 192 == 64 and command & 63 == self._last_command:
            self._send_next_command()
        packet = self._read_packet()
    next_repeat = self.key_repeat.next_event()
    if next_repeat:
        timeout, key = next_repeat
        if not timeout:
            data_input.append(key)
            self.key_repeat.sent_event()
            timeout = None
    return (timeout, data_input, raw_data_input)