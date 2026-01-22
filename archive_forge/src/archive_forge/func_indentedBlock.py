import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def indentedBlock(blockStatementExpr, indentStack, indent=True):
    """Helper method for defining space-delimited indentation blocks, such as
       those used to define block statements in Python source code.

       Parameters:
        - blockStatementExpr - expression defining syntax of statement that
            is repeated within the indented block
        - indentStack - list created by caller to manage indentation stack
            (multiple statementWithIndentedBlock expressions within a single grammar
            should share a common indentStack)
        - indent - boolean indicating whether block must be indented beyond the
            the current level; set to False for block of left-most statements
            (default=True)

       A valid block must contain at least one C{blockStatement}.
    """

    def checkPeerIndent(s, l, t):
        if l >= len(s):
            return
        curCol = col(l, s)
        if curCol != indentStack[-1]:
            if curCol > indentStack[-1]:
                raise ParseFatalException(s, l, 'illegal nesting')
            raise ParseException(s, l, 'not a peer entry')

    def checkSubIndent(s, l, t):
        curCol = col(l, s)
        if curCol > indentStack[-1]:
            indentStack.append(curCol)
        else:
            raise ParseException(s, l, 'not a subentry')

    def checkUnindent(s, l, t):
        if l >= len(s):
            return
        curCol = col(l, s)
        if not (indentStack and curCol < indentStack[-1] and (curCol <= indentStack[-2])):
            raise ParseException(s, l, 'not an unindent')
        indentStack.pop()
    NL = OneOrMore(LineEnd().setWhitespaceChars('\t ').suppress())
    INDENT = Empty() + Empty().setParseAction(checkSubIndent)
    PEER = Empty().setParseAction(checkPeerIndent)
    UNDENT = Empty().setParseAction(checkUnindent)
    if indent:
        smExpr = Group(Optional(NL) + INDENT + OneOrMore(PEER + Group(blockStatementExpr) + Optional(NL)) + UNDENT)
    else:
        smExpr = Group(Optional(NL) + OneOrMore(PEER + Group(blockStatementExpr) + Optional(NL)))
    blockStatementExpr.ignore(_bslash + LineEnd())
    return smExpr