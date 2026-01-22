from io import BytesIO
from fontTools.misc.macRes import ResourceReader, ResourceError
def openTTFonts(path):
    """Given a pathname, return a list of TTFont objects. In the case
    of a flat TTF/OTF file, the list will contain just one font object;
    but in the case of a Mac font suitcase it will contain as many
    font objects as there are sfnt resources in the file.
    """
    from fontTools import ttLib
    fonts = []
    sfnts = getSFNTResIndices(path)
    if not sfnts:
        fonts.append(ttLib.TTFont(path))
    else:
        for index in sfnts:
            fonts.append(ttLib.TTFont(path, index))
        if not fonts:
            raise ttLib.TTLibError("no fonts found in file '%s'" % path)
    return fonts