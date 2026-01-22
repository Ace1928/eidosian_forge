import enum
import shutil
import sys
def writeColor(content, color, end=''):
    forceWrite(f'\x1b[{color}m{content}\x1b[0m', end)