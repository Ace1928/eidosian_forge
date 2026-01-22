from antlr4.Token import Token
def visitChildren(self, node):
    result = self.defaultResult()
    n = node.getChildCount()
    for i in range(n):
        if not self.shouldVisitNextChild(node, result):
            return result
        c = node.getChild(i)
        childResult = c.accept(self)
        result = self.aggregateResult(result, childResult)
    return result