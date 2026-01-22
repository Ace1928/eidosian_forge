from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import re
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_filter
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_transform
import six
Adds self.__key_order_offset to unmarked attribute.order.

      A DFS search that visits each attribute once. The search clears
      skip_reorder attributes marked skip_reorder, otherwise it adds
      self.__key_order_offset to attribute.order.

      Args:
        tree: The attribute subtree to reorder.
      