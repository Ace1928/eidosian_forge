
    Note: This expects kerning to be a flat dictionary
    of kerning pairs, not the nested structure used
    in kerning.plist.

    >>> groups = {
    ...     "public.kern1.O" : ["O", "D", "Q"],
    ...     "public.kern2.E" : ["E", "F"]
    ... }
    >>> kerning = {
    ...     ("public.kern1.O", "public.kern2.E") : -100,
    ...     ("public.kern1.O", "F") : -200,
    ...     ("D", "F") : -300
    ... }
    >>> lookupKerningValue(("D", "F"), kerning, groups)
    -300
    >>> lookupKerningValue(("O", "F"), kerning, groups)
    -200
    >>> lookupKerningValue(("O", "E"), kerning, groups)
    -100
    >>> lookupKerningValue(("O", "O"), kerning, groups)
    0
    >>> lookupKerningValue(("E", "E"), kerning, groups)
    0
    >>> lookupKerningValue(("E", "O"), kerning, groups)
    0
    >>> lookupKerningValue(("X", "X"), kerning, groups)
    0
    >>> lookupKerningValue(("public.kern1.O", "public.kern2.E"),
    ...     kerning, groups)
    -100
    >>> lookupKerningValue(("public.kern1.O", "F"), kerning, groups)
    -200
    >>> lookupKerningValue(("O", "public.kern2.E"), kerning, groups)
    -100
    >>> lookupKerningValue(("public.kern1.X", "public.kern2.X"), kerning, groups)
    0
    