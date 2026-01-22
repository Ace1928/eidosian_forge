import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
def horizontalLine(terminal, y, left, right):
    terminal.selectCharacterSet(insults.CS_DRAWING, insults.G0)
    terminal.cursorPosition(left, y)
    terminal.write(b'q' * (right - left))
    terminal.selectCharacterSet(insults.CS_US, insults.G0)