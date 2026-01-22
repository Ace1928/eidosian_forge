import re
from rdflib.graph import Graph
from rdflib.store import Store
def regexCompareQuad(quad, regexQuad):
    for index in range(4):
        if isinstance(regexQuad[index], REGEXTerm) and (not regexQuad[index].compiledExpr.match(quad[index])):
            return False
    return True