import os, time, webbrowser
from .gui import *
from . import smooth
from .vertex import Vertex
from .arrow import Arrow
from .crossings import Crossing, ECrossing
from .colors import Palette
from .dialog import InfoDialog
from .manager import LinkManager
from .viewer import LinkViewer
from .version import version
from .ipython_tools import IPythonTkRoot
def single_click(self, event):
    """
        Event handler for mouse clicks.
        """
    if self.style_var.get() == 'smooth':
        return
    if self.state == 'start_state':
        if not self.has_focus:
            return
    else:
        self.has_focus = True
    x = self.canvas.canvasx(event.x)
    y = self.canvas.canvasy(event.y)
    self.clear_text()
    start_vertex = Vertex(x, y, self.canvas, style='hidden')
    if self.state == 'start_state':
        if start_vertex in self.Vertices:
            self.state = 'dragging_state'
            self.hide_DT()
            self.hide_labels()
            self.update_info()
            self.canvas.config(cursor=closed_hand_cursor)
            self.ActiveVertex = self.Vertices[self.Vertices.index(start_vertex)]
            self.ActiveVertex.freeze()
            self.saved_crossing_data = self.active_crossing_data()
            x1, y1 = self.ActiveVertex.point()
            if self.ActiveVertex.in_arrow is None and self.ActiveVertex.out_arrow is None:
                self.double_click(event)
                return
            if self.ActiveVertex.in_arrow:
                x0, y0 = self.ActiveVertex.in_arrow.start.point()
                self.ActiveVertex.in_arrow.freeze()
                self.LiveArrow1 = self.canvas.create_line(x0, y0, x1, y1, fill='red')
            if self.ActiveVertex.out_arrow:
                x0, y0 = self.ActiveVertex.out_arrow.end.point()
                self.ActiveVertex.out_arrow.freeze()
                self.LiveArrow2 = self.canvas.create_line(x0, y0, x1, y1, fill='red')
            if self.lock_var.get():
                self.attach_cursor('start')
            return
        elif self.lock_var.get():
            return
        elif start_vertex in self.CrossPoints:
            crossing = self.Crossings[self.CrossPoints.index(start_vertex)]
            if crossing.is_virtual:
                crossing.is_virtual = False
            else:
                crossing.reverse()
            self.update_info()
            crossing.under.draw(self.Crossings)
            crossing.over.draw(self.Crossings)
            self.update_smooth()
            return
        elif self.clicked_on_arrow(start_vertex):
            return
        elif not self.generic_vertex(start_vertex):
            start_vertex.erase()
            self.alert()
            return
        x1, y1 = start_vertex.point()
        start_vertex.set_color(self.palette.new())
        self.Vertices.append(start_vertex)
        self.ActiveVertex = start_vertex
        self.goto_drawing_state(x1, y1)
        return
    elif self.state == 'drawing_state':
        next_vertex = Vertex(x, y, self.canvas, style='hidden')
        if next_vertex == self.ActiveVertex:
            next_vertex.erase()
            dead_arrow = self.ActiveVertex.out_arrow
            if dead_arrow:
                self.destroy_arrow(dead_arrow)
            self.goto_start_state()
            return
        if self.ActiveVertex.out_arrow:
            next_arrow = self.ActiveVertex.out_arrow
            next_arrow.set_end(next_vertex)
            next_vertex.in_arrow = next_arrow
            if not next_arrow.frozen:
                next_arrow.hide()
        else:
            this_color = self.ActiveVertex.color
            next_arrow = Arrow(self.ActiveVertex, next_vertex, self.canvas, style='hidden', color=this_color)
            self.Arrows.append(next_arrow)
        next_vertex.set_color(next_arrow.color)
        if next_vertex in [v for v in self.Vertices if v.is_endpoint()]:
            if not self.generic_arrow(next_arrow):
                self.alert()
                return
            next_vertex.erase()
            next_vertex = self.Vertices[self.Vertices.index(next_vertex)]
            if next_vertex.in_arrow:
                next_vertex.reverse_path()
            next_arrow.set_end(next_vertex)
            next_vertex.in_arrow = next_arrow
            if next_vertex.color != self.ActiveVertex.color:
                self.palette.recycle(self.ActiveVertex.color)
                next_vertex.recolor_incoming(color=next_vertex.color)
            self.update_crossings(next_arrow)
            next_arrow.expose(self.Crossings)
            self.goto_start_state()
            return
        if not (self.generic_vertex(next_vertex) and self.generic_arrow(next_arrow)):
            self.alert()
            self.destroy_arrow(next_arrow)
            return
        self.update_crossings(next_arrow)
        self.update_crosspoints()
        next_arrow.expose(self.Crossings)
        self.Vertices.append(next_vertex)
        next_vertex.expose()
        self.ActiveVertex = next_vertex
        self.canvas.coords(self.LiveArrow1, x, y, x, y)
        return
    elif self.state == 'dragging_state':
        try:
            self.end_dragging_state()
        except ValueError:
            self.alert()