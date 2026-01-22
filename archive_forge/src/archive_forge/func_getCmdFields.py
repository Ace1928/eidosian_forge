from pyparsing import alphas,nums, dblQuotedString, Combine, Word, Group, delimitedList, Suppress, removeQuotes
import string
def getCmdFields(s, l, t):
    t['method'], t['requestURI'], t['protocolVersion'] = t[0].strip('"').split()