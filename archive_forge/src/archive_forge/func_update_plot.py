from pyqtgraph.Qt import QtGui, QtWidgets
import numpy as np
import pyqtgraph as pg
def update_plot():
    pw2.clear()
    nb_glow_lines = params.child('nb glow lines').value()
    alpha_start = params.child('alpha_start').value()
    alpha_stop = params.child('alpha_stop').value()
    alpha_underglow = params.child('alpha_underglow').value()
    linewidth_start = params.child('linewidth_start').value()
    linewidth_stop = params.child('linewidth_stop').value()
    nb_lines = params.child('nb_lines').value()
    xs = []
    ys = []
    for i in range(nb_lines):
        xs.append(np.linspace(0, 2 * np.pi, 100) - i)
        ys.append(np.sin(xs[-1]) * xs[-1] - i / 3 + noises[i])
    for color, x, y in zip(colors, xs, ys):
        pen = pg.mkPen(color=color)
        if params.child('add_underglow').value() == 'Full':
            kw = {'fillLevel': 0.0, 'fillBrush': pg.mkBrush(color='{}{:02x}'.format(color, alpha_underglow))}
        elif params.child('add_underglow').value() == 'Gradient':
            grad = QtGui.QLinearGradient(x.mean(), y.min(), x.mean(), y.max())
            grad.setColorAt(0.001, pg.mkColor(color))
            grad.setColorAt(abs(y.min()) / (y.max() - y.min()), pg.mkColor('{}{:02x}'.format(color, alpha_underglow)))
            grad.setColorAt(0.999, pg.mkColor(color))
            brush = QtGui.QBrush(grad)
            kw = {'fillLevel': 0.0, 'fillBrush': brush}
        else:
            kw = {}
        pw2.addItem(pg.PlotDataItem(x, y, pen=pen, **kw))
        if params.child('make_line_glow').value():
            alphas = np.linspace(alpha_start, alpha_stop, nb_glow_lines, dtype=int)
            lws = np.linspace(linewidth_start, linewidth_stop, nb_glow_lines)
            for alpha, lw in zip(alphas, lws):
                pen = pg.mkPen(color='{}{:02x}'.format(color, alpha), width=lw, connect='finite')
                pw2.addItem(pg.PlotDataItem(x, y, pen=pen))