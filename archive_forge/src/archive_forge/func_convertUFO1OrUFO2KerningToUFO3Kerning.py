def convertUFO1OrUFO2KerningToUFO3Kerning(kerning, groups, glyphSet=()):
    firstReferencedGroups, secondReferencedGroups = findKnownKerningGroups(groups)
    for first, seconds in list(kerning.items()):
        if first in groups and first not in glyphSet:
            if not first.startswith('public.kern1.'):
                firstReferencedGroups.add(first)
        for second in list(seconds.keys()):
            if second in groups and second not in glyphSet:
                if not second.startswith('public.kern2.'):
                    secondReferencedGroups.add(second)
    firstRenamedGroups = {}
    for first in firstReferencedGroups:
        existingGroupNames = list(groups.keys()) + list(firstRenamedGroups.keys())
        newName = first.replace('@MMK_L_', '')
        newName = 'public.kern1.' + newName
        newName = makeUniqueGroupName(newName, existingGroupNames)
        firstRenamedGroups[first] = newName
    secondRenamedGroups = {}
    for second in secondReferencedGroups:
        existingGroupNames = list(groups.keys()) + list(secondRenamedGroups.keys())
        newName = second.replace('@MMK_R_', '')
        newName = 'public.kern2.' + newName
        newName = makeUniqueGroupName(newName, existingGroupNames)
        secondRenamedGroups[second] = newName
    newKerning = {}
    for first, seconds in list(kerning.items()):
        first = firstRenamedGroups.get(first, first)
        newSeconds = {}
        for second, value in list(seconds.items()):
            second = secondRenamedGroups.get(second, second)
            newSeconds[second] = value
        newKerning[first] = newSeconds
    allRenamedGroups = list(firstRenamedGroups.items())
    allRenamedGroups += list(secondRenamedGroups.items())
    for oldName, newName in allRenamedGroups:
        group = list(groups[oldName])
        groups[newName] = group
    return (newKerning, groups, dict(side1=firstRenamedGroups, side2=secondRenamedGroups))