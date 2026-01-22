import gast
import termcolor
def fmt(node, color=True, noanno=False):
    printer = PrettyPrinter(color, noanno)
    if isinstance(node, (list, tuple)):
        for n in node:
            printer.visit(n)
    else:
        printer.visit(node)
    return printer.result