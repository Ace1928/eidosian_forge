import binascii
def getAttributeValue(self, string):
    try:
        return self.attributes[string]
    except KeyError:
        return None