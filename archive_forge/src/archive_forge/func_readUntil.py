import re
def readUntil(self, pattern, include_match=False):
    val = ''
    pattern_match = None
    match_index = self.__position
    if self.hasNext():
        pattern_match = pattern.search(self.__input, self.__position)
        if bool(pattern_match):
            if include_match:
                match_index = pattern_match.end(0)
            else:
                match_index = pattern_match.start(0)
        else:
            match_index = self.__input_length
        val = self.__input[self.__position:match_index]
        self.__position = match_index
    return val