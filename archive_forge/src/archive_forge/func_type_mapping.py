from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Type, Union
from wandb import util
@staticmethod
def type_mapping() -> 'TypeMappingType':
    """Return a map from `_log_type` to subclass. Used to lookup correct types for deserialization.

        Returns:
            dict: dictionary of str:class
        """
    if WBValue._type_mapping is None:
        WBValue._type_mapping = {}
        frontier = [WBValue]
        explored = set()
        while len(frontier) > 0:
            class_option = frontier.pop()
            explored.add(class_option)
            if class_option._log_type is not None:
                WBValue._type_mapping[class_option._log_type] = class_option
            for subclass in class_option.__subclasses__():
                if subclass not in explored:
                    frontier.append(subclass)
    return WBValue._type_mapping