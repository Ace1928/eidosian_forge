import os
import sys
import warnings
from os.path import basename, dirname, exists, join, splitext
from pygame.font import Font
def initsysfonts_win32():
    """initialize fonts dictionary on Windows"""
    fontdir = join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')
    fonts = {}
    microsoft_font_dirs = ['SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Fonts', 'SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Fonts']
    for domain in [_winreg.HKEY_LOCAL_MACHINE, _winreg.HKEY_CURRENT_USER]:
        for font_dir in microsoft_font_dirs:
            try:
                key = _winreg.OpenKey(domain, font_dir)
            except FileNotFoundError:
                continue
            for i in range(_winreg.QueryInfoKey(key)[1]):
                try:
                    name, font, _ = _winreg.EnumValue(key, i)
                except OSError:
                    break
                if splitext(font)[1].lower() not in OpenType_extensions:
                    continue
                if not dirname(font):
                    font = join(fontdir, font)
                for name in name.split('&'):
                    _parse_font_entry_win(name, font, fonts)
    return fonts