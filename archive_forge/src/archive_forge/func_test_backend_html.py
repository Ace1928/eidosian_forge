import pytest
from pybtex.database import parse_string
from pybtex.backends.html import Backend as HtmlBackend
from pybtex.style.formatting.unsrt import Style as UnsrtStyle
@pytest.mark.parametrize('bib,html', [(article_bib, article_html), (book_bib, book_html), (booklet_bib, booklet_html), (inbook_bib, inbook_html), (incollection_bib, incollection_html), (manual_bib, manual_html), (masterthesis_bib, masterthesis_html), (misc_bib, misc_html), (online_bib, online_html), (phdthesis_bib, phdthesis_html), (proceedings_bib, proceedings_html), (techreport_bib, techreport_html), (unpublished_bib, unpublished_html)])
def test_backend_html(bib, html):
    style = UnsrtStyle()
    backend = HtmlBackend()
    bib_data = parse_string(bib, 'bibtex')
    for formatted_entry in style.format_entries(bib_data.entries.values()):
        render = formatted_entry.text.render(backend)
        print(render)
        assert render.strip() == html.strip()