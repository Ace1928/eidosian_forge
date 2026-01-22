from __future__ import absolute_import
from . import Machines
from .Machines import LOWEST_PRIORITY
from .Transitions import TransitionMap
def nfa_to_dfa(old_machine, debug=None):
    """
    Given a nondeterministic Machine, return a new equivalent
    Machine which is deterministic.
    """
    new_machine = Machines.FastMachine()
    state_map = StateMap(new_machine)
    for key, old_state in old_machine.initial_states.items():
        new_state = state_map.old_to_new(epsilon_closure(old_state))
        new_machine.make_initial_state(key, new_state)
    for new_state in new_machine.states:
        transitions = TransitionMap()
        for old_state in state_map.new_to_old(new_state):
            for event, old_target_states in old_state.transitions.items():
                if event and old_target_states:
                    transitions.add_set(event, set_epsilon_closure(old_target_states))
        for event, old_states in transitions.items():
            new_machine.add_transitions(new_state, event, state_map.old_to_new(old_states))
    if debug:
        debug.write('\n===== State Mapping =====\n')
        state_map.dump(debug)
    return new_machine