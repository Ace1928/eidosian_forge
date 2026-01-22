from __future__ import annotations
import copy
import os
import typing as T
from .. import compilers, environment, mesonlib, optinterpreter
from .. import coredata as cdata
from ..build import Executable, Jar, SharedLibrary, SharedModule, StaticLibrary
from ..compilers import detect_compiler_for
from ..interpreterbase import InvalidArguments, SubProject
from ..mesonlib import MachineChoice, OptionKey
from ..mparser import BaseNode, ArithmeticNode, ArrayNode, ElementaryNode, IdNode, FunctionNode, BaseStringNode
from .interpreter import AstInterpreter
def build_target(self, node: BaseNode, args: T.List[TYPE_var], kwargs_raw: T.Dict[str, TYPE_var], targetclass: T.Type[BuildTarget]) -> T.Optional[T.Dict[str, T.Any]]:
    args = self.flatten_args(args)
    if not args or not isinstance(args[0], str):
        return None
    name = args[0]
    srcqueue = [node]
    extra_queue = []
    if 'sources' in kwargs_raw:
        srcqueue += mesonlib.listify(kwargs_raw['sources'])
    if 'extra_files' in kwargs_raw:
        extra_queue += mesonlib.listify(kwargs_raw['extra_files'])
    kwargs = self.flatten_kwargs(kwargs_raw, True)

    def traverse_nodes(inqueue: T.List[BaseNode]) -> T.List[BaseNode]:
        res: T.List[BaseNode] = []
        while inqueue:
            curr = inqueue.pop(0)
            arg_node = None
            assert isinstance(curr, BaseNode)
            if isinstance(curr, FunctionNode):
                arg_node = curr.args
            elif isinstance(curr, ArrayNode):
                arg_node = curr.args
            elif isinstance(curr, IdNode):
                assert isinstance(curr.value, str)
                var_name = curr.value
                if var_name in self.assignments:
                    tmp_node = self.assignments[var_name]
                    if isinstance(tmp_node, (ArrayNode, IdNode, FunctionNode)):
                        inqueue += [tmp_node]
            elif isinstance(curr, ArithmeticNode):
                inqueue += [curr.left, curr.right]
            if arg_node is None:
                continue
            arg_nodes = arg_node.arguments.copy()
            if isinstance(curr, FunctionNode) and curr.func_name.value in BUILD_TARGET_FUNCTIONS:
                arg_nodes.pop(0)
            elementary_nodes = [x for x in arg_nodes if isinstance(x, (str, BaseStringNode))]
            inqueue += [x for x in arg_nodes if isinstance(x, (FunctionNode, ArrayNode, IdNode, ArithmeticNode))]
            if elementary_nodes:
                res += [curr]
        return res
    source_nodes = traverse_nodes(srcqueue)
    extraf_nodes = traverse_nodes(extra_queue)
    kwargs_reduced = {k: v for k, v in kwargs.items() if k in targetclass.known_kwargs and k in {'install', 'build_by_default', 'build_always'}}
    kwargs_reduced = {k: v.value if isinstance(v, ElementaryNode) else v for k, v in kwargs_reduced.items()}
    kwargs_reduced = {k: v for k, v in kwargs_reduced.items() if not isinstance(v, BaseNode)}
    for_machine = MachineChoice.HOST
    objects: T.List[T.Any] = []
    empty_sources: T.List[T.Any] = []
    kwargs_reduced['_allow_no_sources'] = True
    target = targetclass(name, self.subdir, self.subproject, for_machine, empty_sources, None, objects, self.environment, self.coredata.compilers[for_machine], kwargs_reduced)
    target.process_compilers_late()
    new_target = {'name': target.get_basename(), 'id': target.get_id(), 'type': target.get_typename(), 'defined_in': os.path.normpath(os.path.join(self.source_root, self.subdir, environment.build_filename)), 'subdir': self.subdir, 'build_by_default': target.build_by_default, 'installed': target.should_install(), 'outputs': target.get_outputs(), 'sources': source_nodes, 'extra_files': extraf_nodes, 'kwargs': kwargs, 'node': node}
    self.targets += [new_target]
    return new_target