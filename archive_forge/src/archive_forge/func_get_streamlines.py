import math
from plotly import exceptions, optional_imports
from plotly.figure_factory import utils
from plotly.graph_objs import graph_objs
def get_streamlines(self):
    """
        Get streamlines by building trajectory set.
        """
    for indent in range(self.density // 2):
        for xi in range(self.density - 2 * indent):
            self.traj(xi + indent, indent)
            self.traj(xi + indent, self.density - 1 - indent)
            self.traj(indent, xi + indent)
            self.traj(self.density - 1 - indent, xi + indent)
    self.st_x = [np.array(t[0]) * self.delta_x + self.x[0] for t in self.trajectories]
    self.st_y = [np.array(t[1]) * self.delta_y + self.y[0] for t in self.trajectories]
    for index in range(len(self.st_x)):
        self.st_x[index] = self.st_x[index].tolist()
        self.st_x[index].append(np.nan)
    for index in range(len(self.st_y)):
        self.st_y[index] = self.st_y[index].tolist()
        self.st_y[index].append(np.nan)