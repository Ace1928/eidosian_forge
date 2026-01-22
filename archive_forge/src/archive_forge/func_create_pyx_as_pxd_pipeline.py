from __future__ import absolute_import
import itertools
from time import time
from . import Errors
from . import DebugFlags
from . import Options
from .Errors import CompileError, InternalError, AbortError
from . import Naming
def create_pyx_as_pxd_pipeline(context, result):
    from .ParseTreeTransforms import AlignFunctionDefinitions, MarkClosureVisitor, WithTransform, AnalyseDeclarationsTransform
    from .Optimize import ConstantFolding, FlattenInListTransform
    from .Nodes import StatListNode
    pipeline = []
    pyx_pipeline = create_pyx_pipeline(context, context.options, result, exclude_classes=[AlignFunctionDefinitions, MarkClosureVisitor, ConstantFolding, FlattenInListTransform, WithTransform])
    from .Visitor import VisitorTransform

    class SetInPxdTransform(VisitorTransform):

        def visit_StatNode(self, node):
            if hasattr(node, 'in_pxd'):
                node.in_pxd = True
            self.visitchildren(node)
            return node
        visit_Node = VisitorTransform.recurse_to_children
    for stage in pyx_pipeline:
        pipeline.append(stage)
        if isinstance(stage, AnalyseDeclarationsTransform):
            pipeline.insert(-1, SetInPxdTransform())
            break

    def fake_pxd(root):
        for entry in root.scope.entries.values():
            if not entry.in_cinclude:
                entry.defined_in_pxd = 1
                if entry.name == entry.cname and entry.visibility != 'extern':
                    entry.cname = entry.scope.mangle(Naming.func_prefix, entry.name)
        return (StatListNode(root.pos, stats=[]), root.scope)
    pipeline.append(fake_pxd)
    return pipeline