def parse_keywords(f):
    kw = []
    for m in re.finditer('\\s*<entry><token>([^<]+)</token></entry>\\s*<entry>([^<]+)</entry>', f.read()):
        kw.append(m.group(1))
    if not kw:
        raise ValueError('no keyword found')
    kw.sort()
    return kw