import sys
import pytest
import itertools
from threading import Thread
from kivy.tests.common import GraphicUnitTest, requires_graphics
def test_antialiasing_line(self):
    from kivy.uix.widget import Widget
    from kivy.graphics import Color, Rectangle, Instruction
    from kivy.graphics.vertex_instructions import AntiAliasingLine
    r = self.render
    with pytest.raises(TypeError):
        AntiAliasingLine(None, points=[10, 20, 30, 20, 30, 10])
    target_rect = Rectangle()
    AntiAliasingLine(target_rect, points=[10, 20, 30, 40, 50, 60])
    pixels = b'\xff\xff\xff\x00\xff\xff\xff\xff\xff\xff\xff\x00'
    instruction = Instruction()
    aa_line = AntiAliasingLine(instruction)
    assert aa_line.texture.pixels == pixels
    assert aa_line.width == 2.5
    points_1 = [51.0, 649.0, 199.0, 649.0, 199.0, 501.0, 51.0, 501.0]
    points_2 = [261.0, 275.0, 335.0, 349.0, 335.0, 349.0, 409.0, 275.0, 409.0, 275.0, 335.0, 201.0, 335.0, 201.0, 261.0, 275.0]
    points_3 = [260.0, 275.0, 261.0, 275.0, 261.0, 275.0, 261.999999999999, 275.99999999, 261.06667650085353, 278.14064903651496, 261.26658584785304, 281.2756384111877, 261.56658584785305, 281.3756384111877, 261.5993677908431, 284.39931866126904, 262.0644226342696, 287.50606070381684, 262.0644226342696, 287.50606070381684, 262.6609123178712, 290.59026597968375, 263.3877619269211, 293.6463765424993, 264.2436616292954, 296.66888507446475, 265.22706903587977, 299.65234481091227, 265.22706903587977, 299.65234481091227, 266.3362119800583, 302.59137935574284, 267.5690917112779, 305.48069237005546, 268.9234864969319, 308.31507711650784, 270.39695562607204, 311.089425842209, 270.89695562607204, 311.589425842209, 271.98684380773494, 313.7987389832352, 273.69028595595563, 316.4381341741821, 275.50421235284637, 319.00285504651725, 275.50421235284637, 319.00285504651725, 277.4253541804354, 321.48827979987755, 279.45024941129833, 323.8899295308661, 281.57524904736516, 326.20347630433844, 283.79652369566156, 328.4247509526349, 283.99652369566155, 328.7247509526349, 286.1100704691339, 330.54975058870167, 288.5117202001224, 332.5746458195646, 288.5117202001224, 332.5746458195646, 290.99714495348275, 334.4957876471537, 293.5618658258179, 336.3097140440444, 293.5618658258179, 336.3097140440444, 293.2618658258179, 336.1097140440444]
    points_4 = [100, 100, 200, 100]
    for points in (points_1, points_2, points_3, points_4):
        wid = Widget()
        with wid.canvas:
            Color(1, 1, 1, 0.5)
            inst = Instruction()
            aa_line = AntiAliasingLine(inst, points=points)
        r(wid)
        filtered_points = self._filtered_points(points)
        assert aa_line.points == filtered_points + filtered_points[:2]
    for points in (points_1, points_2, points_3, points_4):
        wid = Widget()
        with wid.canvas:
            Color(1, 1, 1, 0.5)
            inst = Instruction()
            aa_line = AntiAliasingLine(inst, points=points, close=0)
        r(wid)
        assert aa_line.points == self._filtered_points(points)