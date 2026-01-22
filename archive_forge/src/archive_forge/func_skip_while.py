import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def skip_while(self, test_set, include=True):
    string = self.string
    pos = self.pos
    try:
        if self.ignore_space:
            while True:
                if string[pos].isspace():
                    pos += 1
                elif string[pos] == '#':
                    pos = string.index('\n', pos)
                elif (string[pos] in test_set) == include:
                    pos += 1
                else:
                    break
        else:
            while (string[pos] in test_set) == include:
                pos += 1
        self.pos = pos
    except IndexError:
        self.pos = len(string)
    except ValueError:
        self.pos = len(string)