def maxCtxFont(font):
    """Calculate the usMaxContext value for an entire font."""
    maxCtx = 0
    for tag in ('GSUB', 'GPOS'):
        if tag not in font:
            continue
        table = font[tag].table
        if not table.LookupList:
            continue
        for lookup in table.LookupList.Lookup:
            for st in lookup.SubTable:
                maxCtx = maxCtxSubtable(maxCtx, tag, lookup.LookupType, st)
    return maxCtx