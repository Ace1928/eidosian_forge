import json
import re
import debtcollector
from oslo_utils._i18n import _
from oslo_utils import strutils
Parse Qemu image information from command `qemu-img info`'s output.

    The instance of :class:`QemuImgInfo` has properties: `image`,
    `backing_file`, `file_format`, `virtual_size`, `cluster_size`,
    `disk_size`, `snapshots` and `encrypted`.

    The parameter format can be set to 'json' or 'human'. With 'json' format
    output, qemu image information will be parsed more easily and readable.
    However 'human' format support will be dropped in next cycle and only
    'json' format will be supported. Prefer to use 'json' instead of 'human'.
    