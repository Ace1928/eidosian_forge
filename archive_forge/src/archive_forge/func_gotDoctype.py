from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def gotDoctype(self, doctype):
    """Encountered DOCTYPE

        This is really grotty: it basically just gives you everything between
        '<!DOCTYPE' and '>' as an argument.
        """
    print('!DOCTYPE', repr(doctype))