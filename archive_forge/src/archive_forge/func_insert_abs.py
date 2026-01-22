import codecs
import copy
import sys
import warnings
def insert_abs(self, r, c, ch):
    """This inserts a character at (r,c). Everything under
        and to the right is shifted right one character.
        The last character of the line is lost.
        """
    if isinstance(ch, bytes):
        ch = self._decode(ch)
    r = constrain(r, 1, self.rows)
    c = constrain(c, 1, self.cols)
    for ci in range(self.cols, c, -1):
        self.put_abs(r, ci, self.get_abs(r, ci - 1))
    self.put_abs(r, c, ch)