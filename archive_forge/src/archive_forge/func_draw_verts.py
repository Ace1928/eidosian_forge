import pyglet.gl as pgl
from sympy.core import S
from sympy.plotting.pygletplot.plot_mode_base import PlotModeBase
def draw_verts(self, use_cverts, use_solid_color):

    def f():
        for u in range(1, len(self.u_set)):
            pgl.glBegin(pgl.GL_QUAD_STRIP)
            for v in range(len(self.v_set)):
                pa = self.verts[u - 1][v]
                pb = self.verts[u][v]
                if pa is None or pb is None:
                    pgl.glEnd()
                    pgl.glBegin(pgl.GL_QUAD_STRIP)
                    continue
                if use_cverts:
                    ca = self.cverts[u - 1][v]
                    cb = self.cverts[u][v]
                    if ca is None:
                        ca = (0, 0, 0)
                    if cb is None:
                        cb = (0, 0, 0)
                elif use_solid_color:
                    ca = cb = self.default_solid_color
                else:
                    ca = cb = self.default_wireframe_color
                pgl.glColor3f(*ca)
                pgl.glVertex3f(*pa)
                pgl.glColor3f(*cb)
                pgl.glVertex3f(*pb)
            pgl.glEnd()
    return f