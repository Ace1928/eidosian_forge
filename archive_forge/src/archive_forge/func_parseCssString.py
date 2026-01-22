def parseCssString(str):
    rules = []
    chunks = str.split('}')
    for chunk in chunks:
        bits = chunk.split('{')
        if len(bits) != 2:
            continue
        rule = {}
        rule['selector'] = bits[0].strip()
        bites = bits[1].strip().split(';')
        if len(bites) < 1:
            continue
        props = {}
        for bite in bites:
            nibbles = bite.strip().split(':')
            if len(nibbles) != 2:
                continue
            props[nibbles[0].strip()] = nibbles[1].strip()
        rule['properties'] = props
        rules.append(rule)
    return rules