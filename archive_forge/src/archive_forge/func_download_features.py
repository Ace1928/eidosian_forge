import argparse
import pathlib
from cartopy import config
from cartopy.feature import Feature, GSHHSFeature, NaturalEarthFeature
from cartopy.io import Downloader, DownloadWarning
def download_features(group_names, dry_run=True):
    for group_name in group_names:
        feature_defns = FEATURE_DEFN_GROUPS[group_name]
        if isinstance(feature_defns, Feature):
            feature = feature_defns
            level = list(feature._levels)[0]
            downloader = Downloader.from_config(('shapefiles', 'gshhs', feature._scale, level))
            format_dict = {'config': config, 'scale': feature._scale, 'level': level}
            if dry_run:
                print(f'URL: {downloader.url(format_dict)}')
            else:
                downloader.path(format_dict)
                geoms = list(feature.geometries())
                print(f'Feature {feature} length: {len(geoms)}')
        else:
            for category, name, scales in feature_defns:
                if not isinstance(scales, tuple):
                    scales = (scales,)
                for scale in scales:
                    downloader = Downloader.from_config(('shapefiles', 'natural_earth', scale, category, name))
                    feature = NaturalEarthFeature(category, name, scale)
                    format_dict = {'config': config, 'category': category, 'name': name, 'resolution': scale}
                    if dry_run:
                        print(f'URL: {downloader.url(format_dict)}')
                    else:
                        downloader.path(format_dict)
                        geoms = list(feature.geometries())
                        print('Feature {}, {}, {} length: {}'.format(category, name, scale, len(geoms)))