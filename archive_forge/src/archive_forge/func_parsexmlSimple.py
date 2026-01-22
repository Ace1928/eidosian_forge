def parsexmlSimple(xmltext, oneOutermostTag=0, eoCB=None, entityReplacer=unEscapeContentList):
    """official interface: discard unused cursor info"""
    if RequirePyRXP:
        raise ImportError('pyRXP not found, fallback parser disabled')
    result, cursor = parsexml0(xmltext, entityReplacer=entityReplacer)
    if oneOutermostTag:
        return result[2][0]
    else:
        return result