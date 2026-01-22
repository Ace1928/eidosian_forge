import json
import os
import click
from rasterio.errors import FileOverwriteError
def write_features(fobj, collection, sequence=False, geojson_type='feature', use_rs=False, **dump_kwds):
    """Read an iterator of (feat, bbox) pairs and write to file using
    the selected modes."""
    if sequence:
        for feat in collection():
            xs, ys = zip(*coords(feat))
            bbox = (min(xs), min(ys), max(xs), max(ys))
            if use_rs:
                fobj.write(u'\x1e')
            if geojson_type == 'bbox':
                fobj.write(json.dumps(bbox, **dump_kwds))
            else:
                fobj.write(json.dumps(feat, **dump_kwds))
            fobj.write('\n')
    else:
        features = list(collection())
        if geojson_type == 'bbox':
            fobj.write(json.dumps(collection.bbox, **dump_kwds))
        else:
            fobj.write(json.dumps({'bbox': collection.bbox, 'type': 'FeatureCollection', 'features': features}, **dump_kwds))
        fobj.write('\n')