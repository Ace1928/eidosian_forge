def findKnownKerningGroups(groups):
    """
    This will find kerning groups with known prefixes.
    In some cases not all kerning groups will be referenced
    by the kerning pairs. The algorithm for locating groups
    in convertUFO1OrUFO2KerningToUFO3Kerning will miss these
    unreferenced groups. By scanning for known prefixes
    this function will catch all of the prefixed groups.

    These are the prefixes and sides that are handled:
    @MMK_L_ - side 1
    @MMK_R_ - side 2

    >>> testGroups = {
    ...     "@MMK_L_1" : None,
    ...     "@MMK_L_2" : None,
    ...     "@MMK_L_3" : None,
    ...     "@MMK_R_1" : None,
    ...     "@MMK_R_2" : None,
    ...     "@MMK_R_3" : None,
    ...     "@MMK_l_1" : None,
    ...     "@MMK_r_1" : None,
    ...     "@MMK_X_1" : None,
    ...     "foo" : None,
    ... }
    >>> first, second = findKnownKerningGroups(testGroups)
    >>> sorted(first) == ['@MMK_L_1', '@MMK_L_2', '@MMK_L_3']
    True
    >>> sorted(second) == ['@MMK_R_1', '@MMK_R_2', '@MMK_R_3']
    True
    """
    knownFirstGroupPrefixes = ['@MMK_L_']
    knownSecondGroupPrefixes = ['@MMK_R_']
    firstGroups = set()
    secondGroups = set()
    for groupName in list(groups.keys()):
        for firstPrefix in knownFirstGroupPrefixes:
            if groupName.startswith(firstPrefix):
                firstGroups.add(groupName)
                break
        for secondPrefix in knownSecondGroupPrefixes:
            if groupName.startswith(secondPrefix):
                secondGroups.add(groupName)
                break
    return (firstGroups, secondGroups)