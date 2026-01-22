import os
import mercantile
import pytest
import requests
import xyzservices.providers as xyz
def get_tile(provider):
    bounds = provider.get('bounds', [[-180, -90], [180, 90]])
    lat = (bounds[0][0] + bounds[1][0]) / 2
    lon = (bounds[0][1] + bounds[1][1]) / 2
    zoom = (provider.get('min_zoom', 0) + provider.get('max_zoom', 20)) // 2
    tile = mercantile.tile(lon, lat, zoom)
    z = tile.z
    x = tile.x
    y = tile.y
    return (z, x, y)