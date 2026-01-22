from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def do_tagstart(self, byte):
    if byte.isalnum() or byte in identChars:
        self.tagName += byte
        if self.tagName == '!--':
            return 'comment'
    elif byte.isspace():
        if self.tagName:
            if self.endtag:
                return 'waitforgt'
            return 'attrs'
        else:
            self._parseError('Whitespace before tag-name')
    elif byte == '>':
        if self.endtag:
            self.gotTagEnd(self.tagName)
            return 'bodydata'
        else:
            self.gotTagStart(self.tagName, {})
            return not self.beExtremelyLenient and 'bodydata' or self.maybeBodyData()
    elif byte == '/':
        if self.tagName:
            return 'afterslash'
        else:
            self.endtag = 1
    elif byte in '!?':
        if self.tagName:
            if not self.beExtremelyLenient:
                self._parseError('Invalid character in tag-name')
        else:
            self.tagName += byte
            self.termtag = 1
    elif byte == '[':
        if self.tagName == '!':
            return 'expectcdata'
        else:
            self._parseError("Invalid '[' in tag-name")
    else:
        if self.beExtremelyLenient:
            self.bodydata = '<'
            return 'unentity'
        self._parseError('Invalid tag character: %r' % byte)