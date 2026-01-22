import re
def peekUntilAfter(self, pattern):
    start = self.__position
    val = self.readUntilAfter(pattern)
    self.__position = start
    return val