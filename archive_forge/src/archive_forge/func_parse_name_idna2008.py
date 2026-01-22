def parse_name_idna2008(name):
    parts = name.split('.')
    r = []
    for part in parts:
        if is_all_ascii(part):
            r.append(part.encode('ascii'))
        else:
            r.append(idna2008.encode(part))
    return b'.'.join(r)