from abc import ABCMeta, abstractmethod
import concurrent.futures
import io
from pathlib import Path
import warnings
import numpy as np
from PIL import Image
import shapely.geometry as sgeom
import cartopy
import cartopy.crs as ccrs
def image_for_domain(self, target_domain, target_z):
    tiles = []

    def fetch_tile(tile):
        try:
            img, extent, origin = self.get_image(tile)
        except OSError:
            raise
        img = np.array(img)
        x = np.linspace(extent[0], extent[1], img.shape[1])
        y = np.linspace(extent[2], extent[3], img.shape[0])
        return (img, x, y, origin)
    with concurrent.futures.ThreadPoolExecutor(max_workers=self._MAX_THREADS) as executor:
        futures = []
        for tile in self.find_images(target_domain, target_z):
            futures.append(executor.submit(fetch_tile, tile))
        for future in concurrent.futures.as_completed(futures):
            try:
                img, x, y, origin = future.result()
                tiles.append([img, x, y, origin])
            except OSError:
                pass
    img, extent, origin = _merge_tiles(tiles)
    return (img, extent, origin)