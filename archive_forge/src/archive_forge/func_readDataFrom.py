import codecs
from antlr4.InputStream import InputStream
def readDataFrom(self, fileName: str, encoding: str, errors: str='strict'):
    with open(fileName, 'rb') as file:
        bytes = file.read()
        return codecs.decode(bytes, encoding, errors)