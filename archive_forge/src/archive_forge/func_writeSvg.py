def writeSvg(self, write):
    if not self.formatted:
        self.format()
    return DiagramItem.writeSvg(self, write)