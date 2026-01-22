import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
from mpl_toolkits.mplot3d.art3d import Line3DCollection
def test_zordered_error():
    lc = [(np.fromiter([0.0, 0.0, 0.0], dtype='float'), np.fromiter([1.0, 1.0, 1.0], dtype='float'))]
    pc = [np.fromiter([0.0, 0.0], dtype='float'), np.fromiter([0.0, 1.0], dtype='float'), np.fromiter([1.0, 1.0], dtype='float')]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.add_collection(Line3DCollection(lc))
    ax.scatter(*pc, visible=False)
    plt.draw()