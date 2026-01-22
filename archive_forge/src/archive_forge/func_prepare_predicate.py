import re
def prepare_predicate(next, token):
    signature = ''
    predicate = []
    while 1:
        token = next()
        if token[0] == ']':
            break
        if token == ('', ''):
            continue
        if token[0] and token[0][:1] in '\'"':
            token = ("'", token[0][1:-1])
        signature += token[0] or '-'
        predicate.append(token[1])
    if signature == '@-':
        key = predicate[1]

        def select(result):
            for elem in result:
                if elem.get(key) is not None:
                    yield elem
        return select
    if signature == "@-='":
        key = predicate[1]
        value = predicate[-1]

        def select(result):
            for elem in result:
                if elem.get(key) == value:
                    yield elem
        return select
    if signature == '-' and (not re.match('-?\\d+$', predicate[0])):
        tag = predicate[0]

        def select(result):
            for elem in result:
                for _ in elem.iterchildren(tag):
                    yield elem
                    break
        return select
    if signature == ".='" or (signature == "-='" and (not re.match('-?\\d+$', predicate[0]))):
        tag = predicate[0]
        value = predicate[-1]
        if tag:

            def select(result):
                for elem in result:
                    for e in elem.iterchildren(tag):
                        if ''.join(e.itertext()) == value:
                            yield elem
                            break
        else:

            def select(result):
                for elem in result:
                    if ''.join(elem.itertext()) == value:
                        yield elem
        return select
    if signature == '-' or signature == '-()' or signature == '-()-':
        if signature == '-':
            index = int(predicate[0]) - 1
            if index < 0:
                if index == -1:
                    raise SyntaxError('indices in path predicates are 1-based, not 0-based')
                else:
                    raise SyntaxError('path index >= 1 expected')
        else:
            if predicate[0] != 'last':
                raise SyntaxError('unsupported function')
            if signature == '-()-':
                try:
                    index = int(predicate[2]) - 1
                except ValueError:
                    raise SyntaxError('unsupported expression')
            else:
                index = -1

        def select(result):
            for elem in result:
                parent = elem.getparent()
                if parent is None:
                    continue
                try:
                    elems = list(parent.iterchildren(elem.tag))
                    if elems[index] is elem:
                        yield elem
                except IndexError:
                    pass
        return select
    raise SyntaxError('invalid predicate')