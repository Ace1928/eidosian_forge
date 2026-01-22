import re
def select_negated(context, result):
    for elem in result:
        if ''.join(elem.itertext()) != value:
            yield elem