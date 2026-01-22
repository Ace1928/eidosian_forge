from __future__ import unicode_literals
from pybtex.style import FormattedEntry, FormattedBibliography
from pybtex.style.template import node, join
from pybtex.richtext import Symbol
from pybtex.plugin import Plugin, find_plugin
def format_bibliography(self, bib_data, citations=None):
    """
        Format bibliography entries with the given keys and return a
        ``FormattedBibliography`` object.

        :param bib_data: A :py:class:`pybtex.database.BibliographyData` object.
        :param citations: A list of citation keys.
        """
    if citations is None:
        citations = list(bib_data.entries.keys())
    citations = bib_data.add_extra_citations(citations, self.min_crossrefs)
    entries = [bib_data.entries[key] for key in citations]
    formatted_entries = self.format_entries(entries)
    formatted_bibliography = FormattedBibliography(formatted_entries, style=self, preamble=bib_data.preamble)
    return formatted_bibliography