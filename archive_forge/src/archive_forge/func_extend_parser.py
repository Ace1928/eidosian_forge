import argparse
import operator
import sys
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING, Union
import gitlab
import gitlab.base
import gitlab.v4.objects
from gitlab import cli
from gitlab.exceptions import GitlabCiLintError
def extend_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    subparsers = parser.add_subparsers(title='resource', dest='gitlab_resource', help='The GitLab resource to manipulate.')
    subparsers.required = True
    classes = set()
    for cls in gitlab.v4.objects.__dict__.values():
        if not isinstance(cls, type):
            continue
        if issubclass(cls, gitlab.base.RESTManager):
            if cls._obj_cls is not None:
                classes.add(cls._obj_cls)
    for cls in sorted(classes, key=operator.attrgetter('__name__')):
        arg_name = cli.cls_to_gitlab_resource(cls)
        object_group = subparsers.add_parser(arg_name, formatter_class=cli.VerticalHelpFormatter)
        object_subparsers = object_group.add_subparsers(title='action', dest='resource_action', help='Action to execute on the GitLab resource.')
        _populate_sub_parser_by_class(cls, object_subparsers)
        object_subparsers.required = True
    return parser