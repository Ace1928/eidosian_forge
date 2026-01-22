def find_sync_regions(self):
    """Return list of sync regions, where both descendents match the base.

        Generates a list of (base1, base2, a1, a2, b1, b2).  There is
        always a zero-length sync region at the end of all the files.
        """
    ia = ib = 0
    amatches = self.sequence_matcher(None, self.base, self.a).get_matching_blocks()
    bmatches = self.sequence_matcher(None, self.base, self.b).get_matching_blocks()
    len_a = len(amatches)
    len_b = len(bmatches)
    sl = []
    while ia < len_a and ib < len_b:
        abase, amatch, alen = amatches[ia]
        bbase, bmatch, blen = bmatches[ib]
        i = intersect((abase, abase + alen), (bbase, bbase + blen))
        if i:
            intbase = i[0]
            intend = i[1]
            intlen = intend - intbase
            asub = amatch + (intbase - abase)
            bsub = bmatch + (intbase - bbase)
            aend = asub + intlen
            bend = bsub + intlen
            sl.append((intbase, intend, asub, aend, bsub, bend))
        if abase + alen < bbase + blen:
            ia += 1
        else:
            ib += 1
    intbase = len(self.base)
    abase = len(self.a)
    bbase = len(self.b)
    sl.append((intbase, intbase, abase, abase, bbase, bbase))
    return sl