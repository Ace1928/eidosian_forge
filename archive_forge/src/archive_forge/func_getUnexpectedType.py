from antlr3.constants import INVALID_TOKEN_TYPE
def getUnexpectedType(self):
    """Return the token type or char of the unexpected input element"""
    from antlr3.streams import TokenStream
    from antlr3.tree import TreeNodeStream
    if isinstance(self.input, TokenStream):
        return self.token.type
    elif isinstance(self.input, TreeNodeStream):
        adaptor = self.input.treeAdaptor
        return adaptor.getType(self.node)
    else:
        return self.c