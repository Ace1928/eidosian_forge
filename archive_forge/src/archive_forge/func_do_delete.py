import argparse
import operator
import sys
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING, Union
import gitlab
import gitlab.base
import gitlab.v4.objects
from gitlab import cli
from gitlab.exceptions import GitlabCiLintError
def do_delete(self) -> None:
    if TYPE_CHECKING:
        assert isinstance(self.mgr, gitlab.mixins.DeleteMixin)
        assert isinstance(self.cls._id_attr, str)
    id = self.args.pop(self.cls._id_attr)
    try:
        self.mgr.delete(id, **self.args)
    except Exception as e:
        cli.die('Impossible to destroy object', e)