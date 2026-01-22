from pyparsing import *
import ebnf
def tallyCommentChars(s, l, t):
    global commentcharcount, commentlocs
    if l not in commentlocs:
        charCount = len(t[0]) - len(list(filter(str.isspace, t[0])))
        commentcharcount += charCount
        commentlocs.add(l)
    return (l, t)