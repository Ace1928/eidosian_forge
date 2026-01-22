import re
def parse_marked_section(self, i, report=1):
    rawdata = self.rawdata
    assert rawdata[i:i + 3] == '<![', 'unexpected call to parse_marked_section()'
    sectName, j = self._scan_name(i + 3, i)
    if j < 0:
        return j
    if sectName in {'temp', 'cdata', 'ignore', 'include', 'rcdata'}:
        match = _markedsectionclose.search(rawdata, i + 3)
    elif sectName in {'if', 'else', 'endif'}:
        match = _msmarkedsectionclose.search(rawdata, i + 3)
    else:
        raise AssertionError('unknown status keyword %r in marked section' % rawdata[i + 3:j])
    if not match:
        return -1
    if report:
        j = match.start(0)
        self.unknown_decl(rawdata[i + 3:j])
    return match.end(0)