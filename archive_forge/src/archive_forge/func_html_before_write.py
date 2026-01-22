from ebooklib.plugins.base import BasePlugin
from ebooklib.utils import parse_html_string
def html_before_write(self, book, chapter):
    from lxml import etree
    from ebooklib import epub
    try:
        tree = parse_html_string(chapter.content)
    except:
        return
    root = tree.getroottree()
    if len(root.find('body')) != 0:
        body = tree.find('body')
        for footnote in body.xpath('//span[@class="InsertNoteMarker"]'):
            footnote_id = footnote.get('id')[:-8]
            a = footnote.getchildren()[0].getchildren()[0]
            footnote_text = body.xpath('//li[@id="%s"]' % footnote_id)[0]
            a.attrib['{%s}type' % epub.NAMESPACES['EPUB']] = 'noteref'
            ftn = etree.SubElement(body, 'aside', {'id': footnote_id})
            ftn.attrib['{%s}type' % epub.NAMESPACES['EPUB']] = 'footnote'
            ftn_p = etree.SubElement(ftn, 'p')
            ftn_p.text = footnote_text.text
        old_footnote = body.xpath('//ol[@id="InsertNote_NoteList"]')
        if len(old_footnote) > 0:
            body.remove(old_footnote[0])
    chapter.content = etree.tostring(tree, pretty_print=True, encoding='utf-8')