from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def get_transitions(self, trigger='', source='*', dest='*', delegate=False):
    """ Return the transitions from the Machine.
        Args:
            trigger (str): Trigger name of the transition.
            source (str, State or Enum): Limits list to transitions from a certain state.
            dest (str, State or Enum): Limits list to transitions to a certain state.
            delegate (Optional[bool]): If True, consider delegations to parents of source
        Returns:
            list(NestedTransitions): All transitions matching the request.
        """
    with self():
        source_path = [] if source == '*' else source.split(self.state_cls.separator) if isinstance(source, string_types) else self._get_enum_path(source) if isinstance(source, Enum) else self._get_state_path(source)
        dest_path = [] if dest == '*' else dest.split(self.state_cls.separator) if isinstance(dest, string_types) else self._get_enum_path(dest) if isinstance(dest, Enum) else self._get_state_path(dest)
        matches = self.get_nested_transitions(trigger, source_path, dest_path)
        if delegate is False or len(source_path) < 2:
            return matches
        source_path.pop()
        while source_path:
            matches.extend(self.get_transitions(trigger, source=self.state_cls.separator.join(source_path), dest=dest))
            source_path.pop()
        return matches