from reportlab.lib import colors
from reportlab.graphics.shapes import Rect, Drawing, Group, String
from reportlab.graphics.charts.textlabels import Label
from reportlab.graphics.widgetbase import Widget
def getAllTracks(self):
    tracks = []
    for title, speaker, trackId, day, hours, duration in self.data:
        if trackId is not None:
            if trackId not in tracks:
                tracks.append(trackId)
    tracks.sort()
    return tracks