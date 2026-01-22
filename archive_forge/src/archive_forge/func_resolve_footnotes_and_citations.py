import sys
import re
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
def resolve_footnotes_and_citations(self):
    """
        Link manually-labeled footnotes and citations to/from their
        references.
        """
    for footnote in self.document.footnotes:
        for label in footnote['names']:
            if label in self.document.footnote_refs:
                reflist = self.document.footnote_refs[label]
                self.resolve_references(footnote, reflist)
    for citation in self.document.citations:
        for label in citation['names']:
            if label in self.document.citation_refs:
                reflist = self.document.citation_refs[label]
                self.resolve_references(citation, reflist)