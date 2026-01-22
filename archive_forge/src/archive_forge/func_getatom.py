import time, calendar
def getatom(self, atomends=None):
    """Parse an RFC 2822 atom.

        Optional atomends specifies a different set of end token delimiters
        (the default is to use self.atomends).  This is used e.g. in
        getphraselist() since phrase endings must not include the `.' (which
        is legal in phrases)."""
    atomlist = ['']
    if atomends is None:
        atomends = self.atomends
    while self.pos < len(self.field):
        if self.field[self.pos] in atomends:
            break
        else:
            atomlist.append(self.field[self.pos])
        self.pos += 1
    return EMPTYSTRING.join(atomlist)