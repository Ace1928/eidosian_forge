from __future__ import annotations
import abc
import logging
import os
import sys
import typing
import warnings
from urwid import signals
from urwid.util import StoppingContext, int_scale
def register_palette_entry(self, name: str | None, foreground: str, background: str, mono: str | None=None, foreground_high: str | None=None, background_high: str | None=None) -> None:
    """Register a single palette entry.

        name -- new entry/attribute name

        foreground -- a string containing a comma-separated foreground
        color and settings

            Color values:
            'default' (use the terminal's default foreground),
            'black', 'dark red', 'dark green', 'brown', 'dark blue',
            'dark magenta', 'dark cyan', 'light gray', 'dark gray',
            'light red', 'light green', 'yellow', 'light blue',
            'light magenta', 'light cyan', 'white'

            Settings:
            'bold', 'underline', 'blink', 'standout', 'strikethrough'

            Some terminals use 'bold' for bright colors.  Most terminals
            ignore the 'blink' setting.  If the color is not given then
            'default' will be assumed.

        background -- a string containing the background color

            Background color values:
            'default' (use the terminal's default background),
            'black', 'dark red', 'dark green', 'brown', 'dark blue',
            'dark magenta', 'dark cyan', 'light gray'

        mono -- a comma-separated string containing monochrome terminal
        settings (see "Settings" above.)

            None = no terminal settings (same as 'default')

        foreground_high -- a string containing a comma-separated
        foreground color and settings, standard foreground
        colors (see "Color values" above) or high-colors may
        be used

            High-color example values:
            '#009' (0% red, 0% green, 60% red, like HTML colors)
            '#fcc' (100% red, 80% green, 80% blue)
            'g40' (40% gray, decimal), 'g#cc' (80% gray, hex),
            '#000', 'g0', 'g#00' (black),
            '#fff', 'g100', 'g#ff' (white)
            'h8' (color number 8), 'h255' (color number 255)

            None = use foreground parameter value

        background_high -- a string containing the background color,
        standard background colors (see "Background colors" above)
        or high-colors (see "High-color example values" above)
        may be used

            None = use background parameter value
        """
    basic = AttrSpec(foreground, background, 16)
    if isinstance(mono, tuple):
        mono = ','.join(mono)
    if mono is None:
        mono = DEFAULT
    mono_spec = AttrSpec(mono, DEFAULT, 1)
    if foreground_high is None:
        foreground_high = foreground
    if background_high is None:
        background_high = background
    high_256 = AttrSpec(foreground_high, background_high, 256)
    high_true = AttrSpec(foreground_high, background_high, 2 ** 24)

    def large_h(desc: str) -> bool:
        if not desc.startswith('h'):
            return False
        if ',' in desc:
            desc = desc.split(',', 1)[0]
        num = int(desc[1:], 10)
        return num > 15
    if large_h(foreground_high) or large_h(background_high):
        high_88 = basic
    else:
        high_88 = AttrSpec(foreground_high, background_high, 88)
    signals.emit_signal(self, UPDATE_PALETTE_ENTRY, name, basic, mono_spec, high_88, high_256, high_true)
    self._palette[name] = (basic, mono_spec, high_88, high_256, high_true)