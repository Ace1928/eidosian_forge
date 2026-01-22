import time, calendar
def getphraselist(self):
    """Parse a sequence of RFC 2822 phrases.

        A phrase is a sequence of words, which are in turn either RFC 2822
        atoms or quoted-strings.  Phrases are canonicalized by squeezing all
        runs of continuous whitespace into one space.
        """
    plist = []
    while self.pos < len(self.field):
        if self.field[self.pos] in self.FWS:
            self.pos += 1
        elif self.field[self.pos] == '"':
            plist.append(self.getquote())
        elif self.field[self.pos] == '(':
            self.commentlist.append(self.getcomment())
        elif self.field[self.pos] in self.phraseends:
            break
        else:
            plist.append(self.getatom(self.phraseends))
    return plist