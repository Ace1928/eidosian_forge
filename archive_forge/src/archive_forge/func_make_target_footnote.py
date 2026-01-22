import sys
import re
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
def make_target_footnote(self, refuri, refs, notes):
    if refuri in notes:
        footnote = notes[refuri]
        assert len(footnote['names']) == 1
        footnote_name = footnote['names'][0]
    else:
        footnote = nodes.footnote()
        footnote_id = self.document.set_id(footnote)
        footnote_name = 'TARGET_NOTE: ' + footnote_id
        footnote['auto'] = 1
        footnote['names'] = [footnote_name]
        footnote_paragraph = nodes.paragraph()
        footnote_paragraph += nodes.reference('', refuri, refuri=refuri)
        footnote += footnote_paragraph
        self.document.note_autofootnote(footnote)
        self.document.note_explicit_target(footnote, footnote)
    for ref in refs:
        if isinstance(ref, nodes.target):
            continue
        refnode = nodes.footnote_reference(refname=footnote_name, auto=1)
        refnode['classes'] += self.classes
        self.document.note_autofootnote_ref(refnode)
        self.document.note_footnote_ref(refnode)
        index = ref.parent.index(ref) + 1
        reflist = [refnode]
        if not utils.get_trim_footnote_ref_space(self.document.settings):
            if self.classes:
                reflist.insert(0, nodes.inline(text=' ', Classes=self.classes))
            else:
                reflist.insert(0, nodes.Text(' '))
        ref.parent.insert(index, reflist)
    return footnote