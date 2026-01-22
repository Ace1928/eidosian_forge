import codecs
import json
import os
import sys
from typing import (
import wandb
from wandb import util
from wandb.sdk.lib import runid
from wandb.sdk.lib.paths import LogicalPath
from . import _dtypes
from ._private import MEDIA_TMP
from .base_types.media import BatchableMedia
Initializes Object3D from a python object.

        Arguments:
            points (Sequence["Point"]): The points in the point cloud.
            boxes (Sequence["Box3D"]): 3D bounding boxes for labeling the point cloud. Boxes
            are displayed in point cloud visualizations.
            vectors (Optional[Sequence["Vector3D"]]): Each vector is displayed in the point cloud
                visualization. Can be used to indicate directionality of bounding boxes. Defaults to None.
            point_cloud_type ("lidar/beta"): At this time, only the "lidar/beta" type is supported. Defaults to "lidar/beta".
        