import contextlib
from pathlib import Path
from unittest import mock
import pytest
import cartopy
import cartopy.io as cio
from cartopy.io.shapereader import NEShpDownloader
def test_from_config():
    generic_url = 'https://example.com/generic_ne/{name}.zip'
    land_downloader = cio.Downloader(generic_url, '', '')
    generic_ne_downloader = cio.Downloader(generic_url, '', '')
    ocean_spec = ('shapefile', 'natural_earth', '110m', 'physical', 'ocean')
    land_spec = ('shapefile', 'natural_earth', '110m', 'physical', 'land')
    generic_spec = ('shapefile', 'natural_earth')
    target_config = {land_spec: land_downloader, generic_spec: generic_ne_downloader}
    with config_replace(target_config):
        r = cio.Downloader.from_config(ocean_spec)
        assert r.url({'name': 'ocean'}) == 'https://example.com/generic_ne/ocean.zip'
        r = cio.Downloader.from_config(land_spec)
        assert r is land_downloader