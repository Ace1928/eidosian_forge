import argparse
import operator
import sys
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING, Union
import gitlab
import gitlab.base
import gitlab.v4.objects
from gitlab import cli
from gitlab.exceptions import GitlabCiLintError
def do_custom(self) -> Any:
    class_instance: Union[gitlab.base.RESTManager, gitlab.base.RESTObject]
    in_obj = cli.custom_actions[self.cls_name][self.resource_action][2]
    if in_obj:
        data = {}
        if self.mgr._from_parent_attrs:
            for k in self.mgr._from_parent_attrs:
                data[k] = self.parent_args[k]
        if not issubclass(self.cls, gitlab.mixins.GetWithoutIdMixin):
            if TYPE_CHECKING:
                assert isinstance(self.cls._id_attr, str)
            data[self.cls._id_attr] = self.args.pop(self.cls._id_attr)
        class_instance = self.cls(self.mgr, data)
    else:
        class_instance = self.mgr
    method_name = self.resource_action.replace('-', '_')
    return getattr(class_instance, method_name)(**self.args)