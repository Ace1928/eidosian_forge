import sys
import re
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
def resolve_indirect_references(self, target):
    if target.hasattr('refid'):
        attname = 'refid'
        call_method = self.document.note_refid
    elif target.hasattr('refuri'):
        attname = 'refuri'
        call_method = None
    else:
        return
    attval = target[attname]
    for name in target['names']:
        reflist = self.document.refnames.get(name, [])
        if reflist:
            target.note_referenced_by(name=name)
        for ref in reflist:
            if ref.resolved:
                continue
            del ref['refname']
            ref[attname] = attval
            if call_method:
                call_method(ref)
            ref.resolved = 1
            if isinstance(ref, nodes.target):
                self.resolve_indirect_references(ref)
    for id in target['ids']:
        reflist = self.document.refids.get(id, [])
        if reflist:
            target.note_referenced_by(id=id)
        for ref in reflist:
            if ref.resolved:
                continue
            del ref['refid']
            ref[attname] = attval
            if call_method:
                call_method(ref)
            ref.resolved = 1
            if isinstance(ref, nodes.target):
                self.resolve_indirect_references(ref)