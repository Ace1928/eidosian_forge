import re
import _markupbase
from html import unescape
def parse_starttag(self, i):
    self.__starttag_text = None
    endpos = self.check_for_whole_start_tag(i)
    if endpos < 0:
        return endpos
    rawdata = self.rawdata
    self.__starttag_text = rawdata[i:endpos]
    attrs = []
    match = tagfind_tolerant.match(rawdata, i + 1)
    assert match, 'unexpected call to parse_starttag()'
    k = match.end()
    self.lasttag = tag = match.group(1).lower()
    while k < endpos:
        m = attrfind_tolerant.match(rawdata, k)
        if not m:
            break
        attrname, rest, attrvalue = m.group(1, 2, 3)
        if not rest:
            attrvalue = None
        elif attrvalue[:1] == "'" == attrvalue[-1:] or attrvalue[:1] == '"' == attrvalue[-1:]:
            attrvalue = attrvalue[1:-1]
        if attrvalue:
            attrvalue = unescape(attrvalue)
        attrs.append((attrname.lower(), attrvalue))
        k = m.end()
    end = rawdata[k:endpos].strip()
    if end not in ('>', '/>'):
        self.handle_data(rawdata[i:endpos])
        return endpos
    if end.endswith('/>'):
        self.handle_startendtag(tag, attrs)
    else:
        self.handle_starttag(tag, attrs)
        if tag in self.CDATA_CONTENT_ELEMENTS:
            self.set_cdata_mode(tag)
    return endpos