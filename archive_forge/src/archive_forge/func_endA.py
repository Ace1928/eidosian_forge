from reportlab.lib import colors
@property
def endA(self):
    """End position of Feature A."""
    try:
        return self.featureA.end
    except AttributeError:
        track, start, end = self.featureA
        return end