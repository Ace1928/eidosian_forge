from rdflib.graph import Graph
from rdflib.namespace import OWL, Namespace
from rdflib.plugins.serializers.turtle import OBJECT, SUBJECT, TurtleSerializer
def p_clause(self, node, position):
    if isinstance(node, Graph):
        self.subjectDone(node)
        if position is OBJECT:
            self.write(' ')
        self.write('{')
        self.depth += 1
        serializer = N3Serializer(node, parent=self)
        serializer.serialize(self.stream)
        self.depth -= 1
        self.write(self.indent() + '}')
        return True
    else:
        return False