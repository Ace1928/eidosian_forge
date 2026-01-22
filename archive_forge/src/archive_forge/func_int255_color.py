from reportlab.lib import colors
def int255_color(self, values):
    """Map integer (red, green, blue) tuple to a ReportLab Color object.

        - values: A tuple of (red, green, blue) intensities as
          integers in the range 0->255

        Takes a tuple of (red, green, blue) intensity values in the range
        0 -> 255 and returns an appropriate colors.Color object.
        """
    red, green, blue = values
    factor = 1 / 255.0
    red, green, blue = (red * factor, green * factor, blue * factor)
    return colors.Color(red, green, blue)