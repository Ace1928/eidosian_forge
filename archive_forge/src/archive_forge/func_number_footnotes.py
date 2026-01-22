import sys
import re
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
def number_footnotes(self, startnum):
    """
        Assign numbers to autonumbered footnotes.

        For labeled autonumbered footnotes, copy the number over to
        corresponding footnote references.
        """
    for footnote in self.document.autofootnotes:
        while True:
            label = str(startnum)
            startnum += 1
            if label not in self.document.nameids:
                break
        footnote.insert(0, nodes.label('', label))
        for name in footnote['names']:
            for ref in self.document.footnote_refs.get(name, []):
                ref += nodes.Text(label)
                ref.delattr('refname')
                assert len(footnote['ids']) == len(ref['ids']) == 1
                ref['refid'] = footnote['ids'][0]
                footnote.add_backref(ref['ids'][0])
                self.document.note_refid(ref)
                ref.resolved = 1
        if not footnote['names'] and (not footnote['dupnames']):
            footnote['names'].append(label)
            self.document.note_explicit_target(footnote, footnote)
            self.autofootnote_labels.append(label)
    return startnum