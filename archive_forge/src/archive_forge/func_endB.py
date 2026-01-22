from reportlab.lib import colors
@property
def endB(self):
    """End position of Feature B."""
    try:
        return self.featureB.end
    except AttributeError:
        track, start, end = self.featureB
        return end