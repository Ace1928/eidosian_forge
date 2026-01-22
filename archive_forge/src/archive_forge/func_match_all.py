from pythran.passmanager import NodeAnalysis
def match_all(self, *args):
    assert len(args) > 1, 'at least two arguments'
    static = False
    const = True
    for value in args:
        if self.visit(value):
            static = True
        else:
            const &= value in self.constant_expressions
    return static and const