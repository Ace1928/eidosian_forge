from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from collections import OrderedDict
import json
import re
from googlecloudsdk.api_lib.util import resource as resource_util
from googlecloudsdk.core.util import files
import six
Generates parent collections for any existing collection missing one.

    Args:
      collections: [resource.CollectionInfo], The existing collections from the
        discovery doc.
      custom_resources: {str, str}, A mapping of collection name to path that
        have been registered manually in the yaml file.
      api_version: Override api_version for each found resource collection.

    Raises:
      ConflictingCollection: If multiple parent collections have the same name
        but different paths, and a custom resource has not been declared to
        resolve the conflict.

    Returns:
      [resource.CollectionInfo], Additional collections to include in the
      resource module.
    