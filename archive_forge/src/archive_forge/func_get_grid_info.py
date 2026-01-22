import numpy as np
from matplotlib import ticker as mticker
from matplotlib.transforms import Bbox, Transform
def get_grid_info(self, x1, y1, x2, y2):
    """
        lon_values, lat_values : list of grid values. if integer is given,
                           rough number of grids in each direction.
        """
    extremes = self.extreme_finder(self.inv_transform_xy, x1, y1, x2, y2)
    lon_min, lon_max, lat_min, lat_max = extremes
    lon_levs, lon_n, lon_factor = self.grid_locator1(lon_min, lon_max)
    lon_levs = np.asarray(lon_levs)
    lat_levs, lat_n, lat_factor = self.grid_locator2(lat_min, lat_max)
    lat_levs = np.asarray(lat_levs)
    lon_values = lon_levs[:lon_n] / lon_factor
    lat_values = lat_levs[:lat_n] / lat_factor
    lon_lines, lat_lines = self._get_raw_grid_lines(lon_values, lat_values, lon_min, lon_max, lat_min, lat_max)
    ddx = (x2 - x1) * 1e-10
    ddy = (y2 - y1) * 1e-10
    bb = Bbox.from_extents(x1 - ddx, y1 - ddy, x2 + ddx, y2 + ddy)
    grid_info = {'extremes': extremes, 'lon_lines': lon_lines, 'lat_lines': lat_lines, 'lon': self._clip_grid_lines_and_find_ticks(lon_lines, lon_values, lon_levs, bb), 'lat': self._clip_grid_lines_and_find_ticks(lat_lines, lat_values, lat_levs, bb)}
    tck_labels = grid_info['lon']['tick_labels'] = {}
    for direction in ['left', 'bottom', 'right', 'top']:
        levs = grid_info['lon']['tick_levels'][direction]
        tck_labels[direction] = self.tick_formatter1(direction, lon_factor, levs)
    tck_labels = grid_info['lat']['tick_labels'] = {}
    for direction in ['left', 'bottom', 'right', 'top']:
        levs = grid_info['lat']['tick_levels'][direction]
        tck_labels[direction] = self.tick_formatter2(direction, lat_factor, levs)
    return grid_info