from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from typing import Mapping, Sequence
from googlecloudsdk.api_lib.run import k8s_object
@property
def volume_mounts(self):
    """Returns a mutable, dict-like object to manage volume mounts.

    The returned object can be used like a dictionary, and any modifications to
    the returned object (i.e. setting and deleting keys) modify the underlying
    nested volume mounts. There are additional properties on the object
    (e.g. `.secrets` that can be used to access a mutable dict-like object for
    a volume mounts that mount volumes of a given type.
    """
    return VolumeMountsAsDictionaryWrapper(self._volumes, self._m.volumeMounts, self._messages.VolumeMount)