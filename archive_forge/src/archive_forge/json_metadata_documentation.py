import codecs
import os
from typing import TYPE_CHECKING, Type, Union
from wandb import util
from wandb.sdk.lib import runid
from .._private import MEDIA_TMP
from .media import Media
JSONMetadata is a type for encoding arbitrary metadata as files.