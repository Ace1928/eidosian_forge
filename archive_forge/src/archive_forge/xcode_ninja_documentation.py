import errno
import gyp.generator.ninja
import os
import re
import xml.sax.saxutils
Initialize targets for the ninja wrapper.

  This sets up the necessary variables in the targets to generate Xcode projects
  that use ninja as an external builder.
  Arguments:
    target_list: List of target pairs: 'base/base.gyp:base'.
    target_dicts: Dict of target properties keyed on target pair.
    data: Dict of flattened build files keyed on gyp path.
    params: Dict of global options for gyp.
  