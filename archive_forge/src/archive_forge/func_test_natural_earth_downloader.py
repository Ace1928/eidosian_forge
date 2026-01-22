import contextlib
from pathlib import Path
from unittest import mock
import pytest
import cartopy
import cartopy.io as cio
from cartopy.io.shapereader import NEShpDownloader
@pytest.mark.network
def test_natural_earth_downloader(tmp_path):
    shp_path_template = str(tmp_path / '{category}_{resolution}_{name}.shp')
    format_dict = {'category': 'physical', 'name': 'rivers_lake_centerlines', 'resolution': '110m'}
    dnld_item = NEShpDownloader(target_path_template=shp_path_template)
    with mock.patch.object(dnld_item, 'acquire_resource', wraps=dnld_item.acquire_resource) as counter:
        with pytest.warns(cartopy.io.DownloadWarning, match='Downloading:'):
            shp_path = dnld_item.path(format_dict)
    counter.assert_called_once()
    assert shp_path_template.format(**format_dict) == str(shp_path)
    with mock.patch.object(dnld_item, 'acquire_resource', wraps=dnld_item.acquire_resource) as counter:
        assert dnld_item.path(format_dict) == shp_path
    counter.assert_not_called()
    exts = ['.shp', '.shx']
    for ext in exts:
        fname = shp_path.with_suffix(ext)
        assert fname.exists(), f"Shapefile's {ext} file doesn't exist in {fname}"
    pre_dnld = NEShpDownloader(target_path_template='/not/a/real/file.txt', pre_downloaded_path_template=str(shp_path))
    with mock.patch.object(pre_dnld, 'acquire_resource', wraps=pre_dnld.acquire_resource) as counter:
        assert pre_dnld.path(format_dict) == shp_path
    counter.assert_not_called()