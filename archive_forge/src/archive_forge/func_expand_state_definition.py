import keyword
import sys
import os
import types
import importlib
import pyparsing as pp
def expand_state_definition(source, loc, tokens):
    """
    Parse action to convert statemachine to corresponding Python classes and methods
    """
    indent = ' ' * (pp.col(loc, source) - 1)
    statedef = []
    states = set()
    fromTo = {}
    for tn in tokens.transitions:
        states.add(tn.from_state)
        states.add(tn.to_state)
        fromTo[tn.from_state] = tn.to_state
    baseStateClass = tokens.name
    statedef.extend(['class %s(object):' % baseStateClass, '    def __str__(self):', '        return self.__class__.__name__', '    @classmethod', '    def states(cls):', '        return list(cls.__subclasses__())', '    def next_state(self):', '        return self._next_state_class()'])
    statedef.extend(('class {0}({1}): pass'.format(s, baseStateClass) for s in states))
    statedef.extend(('{0}._next_state_class = {1}'.format(s, fromTo[s]) for s in states if s in fromTo))
    statedef.extend(['class {baseStateClass}Mixin:'.format(baseStateClass=baseStateClass), '    def __init__(self):', '        self._state = None', '    def initialize_state(self, init_state):', '        if issubclass(init_state, {baseStateClass}):'.format(baseStateClass=baseStateClass), '            init_state = init_state()', '        self._state = init_state', '    @property', '    def state(self):', '        return self._state', '    # get behavior/properties from current state', '    def __getattr__(self, attrname):', '        attr = getattr(self._state, attrname)', '        return attr', '    def __str__(self):', "       return '{0}: {1}'.format(self.__class__.__name__, self._state)"])
    return ('\n' + indent).join(statedef) + '\n'