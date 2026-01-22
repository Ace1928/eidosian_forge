from fontTools.feaLib.error import FeatureLibError, IncludedFeaNotFound
from fontTools.feaLib.location import FeatureLibLocation
import re
import os
def scan_anonymous_block(self, tag):
    return self.lexers_[-1].scan_anonymous_block(tag)