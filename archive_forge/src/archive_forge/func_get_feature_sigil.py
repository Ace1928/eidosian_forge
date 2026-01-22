from reportlab.graphics.shapes import Drawing, String, Group, Line, Circle, Polygon
from reportlab.lib import colors
from reportlab.graphics.shapes import ArcPath
from ._AbstractDrawer import AbstractDrawer, draw_polygon, intermediate_points
from ._AbstractDrawer import _stroke_and_fill_colors
from ._FeatureSet import FeatureSet
from ._GraphSet import GraphSet
from math import pi, cos, sin
def get_feature_sigil(self, feature, locstart, locend, **kwargs):
    """Return graphics for feature, and any required label for it.

        Arguments:
         - feature       Feature object
         - locstart      The start position of the feature
         - locend        The end position of the feature

        """
    btm, ctr, top = self.track_radii[self.current_track_level]
    startangle, startcos, startsin = self.canvas_angle(locstart)
    endangle, endcos, endsin = self.canvas_angle(locend)
    midangle, midcos, midsin = self.canvas_angle((locend + locstart) / 2)
    draw_methods = {'BOX': self._draw_sigil_box, 'OCTO': self._draw_sigil_cut_corner_box, 'JAGGY': self._draw_sigil_jaggy, 'ARROW': self._draw_sigil_arrow, 'BIGARROW': self._draw_sigil_big_arrow}
    method = draw_methods[feature.sigil]
    kwargs['head_length_ratio'] = feature.arrowhead_length
    kwargs['shaft_height_ratio'] = feature.arrowshaft_height
    if hasattr(feature, 'url'):
        kwargs['hrefURL'] = feature.url
        kwargs['hrefTitle'] = feature.name
    sigil = method(btm, ctr, top, startangle, endangle, feature.location.strand, color=feature.color, border=feature.border, **kwargs)
    if feature.label:
        label = String(0, 0, f' {feature.name.strip()} ', fontName=feature.label_font, fontSize=feature.label_size, fillColor=feature.label_color)
        labelgroup = Group(label)
        if feature.label_strand:
            strand = feature.label_strand
        else:
            strand = feature.location.strand
        if feature.label_position in ('start', "5'", 'left'):
            if strand != -1:
                label_angle = startangle + 0.5 * pi
                sinval, cosval = (startsin, startcos)
            else:
                label_angle = endangle + 0.5 * pi
                sinval, cosval = (endsin, endcos)
        elif feature.label_position in ('middle', 'center', 'centre'):
            label_angle = midangle + 0.5 * pi
            sinval, cosval = (midsin, midcos)
        elif feature.label_position in ('end', "3'", 'right'):
            if strand != -1:
                label_angle = endangle + 0.5 * pi
                sinval, cosval = (endsin, endcos)
            else:
                label_angle = startangle + 0.5 * pi
                sinval, cosval = (startsin, startcos)
        elif startangle < pi:
            label_angle = endangle + 0.5 * pi
            sinval, cosval = (endsin, endcos)
        else:
            label_angle = startangle + 0.5 * pi
            sinval, cosval = (startsin, startcos)
        if strand != -1:
            radius = top
            if startangle < pi:
                label_angle -= pi
            else:
                labelgroup.contents[0].textAnchor = 'end'
        else:
            radius = btm
            if startangle < pi:
                label_angle -= pi
                labelgroup.contents[0].textAnchor = 'end'
        x_pos = self.xcenter + radius * sinval
        y_pos = self.ycenter + radius * cosval
        coslabel = cos(label_angle)
        sinlabel = sin(label_angle)
        labelgroup.transform = (coslabel, -sinlabel, sinlabel, coslabel, x_pos, y_pos)
    else:
        labelgroup = None
    return (sigil, labelgroup)