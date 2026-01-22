def getPyartName(pidfont):
    if not pidfont.bold:
        if pidfont.italic:
            shape = Italic
        else:
            shape = Roman
    elif pidfont.italic:
        shape = BoldItalic
    else:
        shape = Bold
    face = pidfont.face or DefaultFace
    face = face.lower()
    if face in PidLegalFonts:
        return MapPid2PyartFontName[PidLegalFonts[face], shape]
    else:
        raise ValueError('Illegal Font')