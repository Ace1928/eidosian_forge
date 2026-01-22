from collections import Counter
import os
import re
import sys
import traceback
import warnings
from .builder import (
from .dammit import UnicodeDammit
from .element import (
def string_container(self, base_class=None):
    container = base_class or NavigableString
    container = self.element_classes.get(container, container)
    if self.string_container_stack and container is NavigableString:
        container = self.builder.string_containers.get(self.string_container_stack[-1].name, container)
    return container