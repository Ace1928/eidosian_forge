from io import StringIO
from antlr4.error.Errors import IllegalStateException
from antlr4.RuleContext import RuleContext
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNState import ATNState
def getAllContextNodes(context: PredictionContext, nodes: list=None, visited: dict=None):
    if nodes is None:
        nodes = list()
        return getAllContextNodes(context, nodes, visited)
    elif visited is None:
        visited = dict()
        return getAllContextNodes(context, nodes, visited)
    else:
        if context is None or visited.get(context, None) is not None:
            return nodes
        visited.put(context, context)
        nodes.add(context)
        for i in range(0, len(context)):
            getAllContextNodes(context.getParent(i), nodes, visited)
        return nodes