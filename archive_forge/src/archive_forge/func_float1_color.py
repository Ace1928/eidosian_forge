from reportlab.lib import colors
def float1_color(self, values):
    """Map float (red, green, blue) tuple to a ReportLab Color object.

        - values: A tuple of (red, green, blue) intensities as floats
          in the range 0 -> 1

        Takes a tuple of (red, green, blue) intensity values in the range
        0 -> 1 and returns an appropriate colors.Color object.
        """
    red, green, blue = values
    return colors.Color(red, green, blue)