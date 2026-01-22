import sys
import re
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
def circular_indirect_reference(self, target):
    self.indirect_target_error(target, 'forming a circular reference')