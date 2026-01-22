from argparse import (
from gettext import gettext
from typing import Dict, List, Set, Tuple
def take_action(action, argument_strings, option_string=None):
    seen_actions.add(action)
    argument_values = self._get_values(action, argument_strings)
    if argument_values is not action.default:
        seen_non_default_actions.add(action)
        for conflict_action in action_conflicts.get(action, []):
            if conflict_action in seen_non_default_actions:
                msg = gettext('not allowed with argument %s')
                action_name = _get_action_name(conflict_action)
                raise ArgumentError(action, msg % action_name)
    if argument_values is not SUPPRESS or isinstance(action, _SubParsersAction):
        try:
            action(self, namespace, argument_values, option_string)
        except BaseException:
            if isinstance(action, _SubParsersAction):
                subnamespace = action._name_parser_map[argument_values[0]]._argcomplete_namespace
                for key, value in vars(subnamespace).items():
                    setattr(namespace, key, value)
            raise