from pyparsing import *
def convertToHTML(opening, closing):

    def conversionParseAction(s, l, t):
        return opening + t[0] + closing
    return conversionParseAction