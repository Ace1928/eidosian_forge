import keyword
import sys
import os
import types
import importlib
import pyparsing as pp
def expand_named_state_definition(source, loc, tokens):
    """
    Parse action to convert statemachine with named transitions to corresponding Python
    classes and methods
    """
    indent = ' ' * (pp.col(loc, source) - 1)
    statedef = []
    states = set()
    transitions = set()
    baseStateClass = tokens.name
    fromTo = {}
    for tn in tokens.transitions:
        states.add(tn.from_state)
        states.add(tn.to_state)
        transitions.add(tn.transition)
        if tn.from_state in fromTo:
            fromTo[tn.from_state][tn.transition] = tn.to_state
        else:
            fromTo[tn.from_state] = {tn.transition: tn.to_state}
    for s in states:
        if s not in fromTo:
            fromTo[s] = {}
    statedef.extend(['class {baseStateClass}Transition:'.format(baseStateClass=baseStateClass), '    def __str__(self):', '        return self.transitionName'])
    statedef.extend(('{tn_name} = {baseStateClass}Transition()'.format(tn_name=tn, baseStateClass=baseStateClass) for tn in transitions))
    statedef.extend(("{tn_name}.transitionName = '{tn_name}'".format(tn_name=tn) for tn in transitions))
    statedef.extend(['class %s(object):' % baseStateClass, '    from statemachine import InvalidTransitionException as BaseTransitionException', '    class InvalidTransitionException(BaseTransitionException): pass', '    def __str__(self):', '        return self.__class__.__name__', '    @classmethod', '    def states(cls):', '        return list(cls.__subclasses__())', '    @classmethod', '    def next_state(cls, name):', '        try:', '            return cls.tnmap[name]()', '        except KeyError:', "            raise cls.InvalidTransitionException('%s does not support transition %r'% (cls.__name__, name))", '    def __bad_tn(name):', '        def _fn(cls):', "            raise cls.InvalidTransitionException('%s does not support transition %r'% (cls.__name__, name))", '        _fn.__name__ = name', '        return _fn'])
    statedef.extend(('    {tn_name} = classmethod(__bad_tn({tn_name!r}))'.format(tn_name=tn) for tn in transitions))
    statedef.extend(('class %s(%s): pass' % (s, baseStateClass) for s in states))
    for s in states:
        trns = list(fromTo[s].items())
        statedef.extend(('%s.%s = classmethod(lambda cls: %s())' % (s, tn_, to_) for tn_, to_ in trns))
    statedef.extend(['{baseStateClass}.transitions = classmethod(lambda cls: [{transition_class_list}])'.format(baseStateClass=baseStateClass, transition_class_list=', '.join(('cls.{0}'.format(tn) for tn in transitions))), '{baseStateClass}.transition_names = [tn.__name__ for tn in {baseStateClass}.transitions()]'.format(baseStateClass=baseStateClass)])
    statedef.extend(['class {baseStateClass}Mixin:'.format(baseStateClass=baseStateClass), '    def __init__(self):', '        self._state = None', '    def initialize_state(self, init_state):', '        if issubclass(init_state, {baseStateClass}):'.format(baseStateClass=baseStateClass), '            init_state = init_state()', '        self._state = init_state', '    @property', '    def state(self):', '        return self._state', '    # get behavior/properties from current state', '    def __getattr__(self, attrname):', '        attr = getattr(self._state, attrname)', '        return attr', '    def __str__(self):', "       return '{0}: {1}'.format(self.__class__.__name__, self._state)"])
    statedef.extend(('    def {tn_name}(self): self._state = self._state.{tn_name}()'.format(tn_name=tn) for tn in transitions))
    return ('\n' + indent).join(statedef) + '\n'