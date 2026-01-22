import time, calendar
def getdomain(self):
    """Get the complete domain name from an address."""
    sdlist = []
    while self.pos < len(self.field):
        if self.field[self.pos] in self.LWS:
            self.pos += 1
        elif self.field[self.pos] == '(':
            self.commentlist.append(self.getcomment())
        elif self.field[self.pos] == '[':
            sdlist.append(self.getdomainliteral())
        elif self.field[self.pos] == '.':
            self.pos += 1
            sdlist.append('.')
        elif self.field[self.pos] == '@':
            return EMPTYSTRING
        elif self.field[self.pos] in self.atomends:
            break
        else:
            sdlist.append(self.getatom())
    return EMPTYSTRING.join(sdlist)