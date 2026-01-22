from fontTools.feaLib.error import FeatureLibError, IncludedFeaNotFound
from fontTools.feaLib.location import FeatureLibLocation
import re
import os
def location_(self):
    column = self.pos_ - self.line_start_ + 1
    return FeatureLibLocation(self.filename_ or '<features>', self.line_, column)