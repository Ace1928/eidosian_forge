from fontTools.feaLib.error import FeatureLibError, IncludedFeaNotFound
from fontTools.feaLib.location import FeatureLibLocation
import re
import os
def scan_until_(self, stop_at):
    p = self.pos_
    while p < self.text_length_ and self.text_[p] not in stop_at:
        p += 1
    self.pos_ = p