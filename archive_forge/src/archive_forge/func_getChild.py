import binascii
def getChild(self, identifier):
    if type(identifier) == int:
        if len(self.children) > identifier:
            return self.children[identifier]
        else:
            return None
    for c in self.children:
        if identifier == c.tag:
            return c
    return None