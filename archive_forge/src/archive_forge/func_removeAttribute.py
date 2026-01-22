import binascii
def removeAttribute(self, key):
    if key in self.attributes:
        del self.attributes[key]