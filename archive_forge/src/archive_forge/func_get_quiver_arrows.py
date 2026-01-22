import math
from plotly import exceptions
from plotly.graph_objs import graph_objs
from plotly.figure_factory import utils
def get_quiver_arrows(self):
    """
        Creates lists of x and y values to plot the arrows

        Gets length of each barb then calculates the length of each side of
        the arrow. Gets angle of barb and applies angle to each side of the
        arrowhead. Next uses arrow_scale to scale the length of arrowhead and
        creates x and y values for arrowhead point1 and point2. Finally x and y
        values for point1, endpoint and point2s for each arrowhead are
        separated by a None and zipped to create lists of x and y values for
        the arrows.

        :rtype: (list, list) arrow_x, arrow_y: list of point1, endpoint, point2
            x_values separated by a None to create the arrowhead and list of
            point1, endpoint, point2 y_values separated by a None to create
            the barb of the arrow.
        """
    dif_x = [i - j for i, j in zip(self.end_x, self.x)]
    dif_y = [i - j for i, j in zip(self.end_y, self.y)]
    barb_len = [None] * len(self.x)
    for index in range(len(barb_len)):
        barb_len[index] = math.hypot(dif_x[index] / self.scaleratio, dif_y[index])
    arrow_len = [None] * len(self.x)
    arrow_len = [i * self.arrow_scale for i in barb_len]
    barb_ang = [None] * len(self.x)
    for index in range(len(barb_ang)):
        barb_ang[index] = math.atan2(dif_y[index], dif_x[index] / self.scaleratio)
    ang1 = [i + self.angle for i in barb_ang]
    ang2 = [i - self.angle for i in barb_ang]
    cos_ang1 = [None] * len(ang1)
    for index in range(len(ang1)):
        cos_ang1[index] = math.cos(ang1[index])
    seg1_x = [i * j for i, j in zip(arrow_len, cos_ang1)]
    sin_ang1 = [None] * len(ang1)
    for index in range(len(ang1)):
        sin_ang1[index] = math.sin(ang1[index])
    seg1_y = [i * j for i, j in zip(arrow_len, sin_ang1)]
    cos_ang2 = [None] * len(ang2)
    for index in range(len(ang2)):
        cos_ang2[index] = math.cos(ang2[index])
    seg2_x = [i * j for i, j in zip(arrow_len, cos_ang2)]
    sin_ang2 = [None] * len(ang2)
    for index in range(len(ang2)):
        sin_ang2[index] = math.sin(ang2[index])
    seg2_y = [i * j for i, j in zip(arrow_len, sin_ang2)]
    for index in range(len(self.end_x)):
        point1_x = [i - j * self.scaleratio for i, j in zip(self.end_x, seg1_x)]
        point1_y = [i - j for i, j in zip(self.end_y, seg1_y)]
        point2_x = [i - j * self.scaleratio for i, j in zip(self.end_x, seg2_x)]
        point2_y = [i - j for i, j in zip(self.end_y, seg2_y)]
    empty = [None] * len(self.end_x)
    arrow_x = utils.flatten(zip(point1_x, self.end_x, point2_x, empty))
    arrow_y = utils.flatten(zip(point1_y, self.end_y, point2_y, empty))
    return (arrow_x, arrow_y)