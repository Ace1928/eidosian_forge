from builtins import range
import sys
from math import sqrt, cos, sin, atan2, pi
def tikz_draw(self, file, transform):
    points = ['(%.2f, %.2f)' % transform(xy) for xy in self.bezier()]
    file.write(self.color, '    \\draw %s .. controls %s and %s .. ' % tuple(points[:3]))
    for i in range(3, len(points) - 3, 3):
        file.write(self.color, '\n' + 10 * ' ' + '%s .. controls %s and %s .. ' % tuple(points[i:i + 3]))
    file.write(self.color, points[-1] + ';\n')