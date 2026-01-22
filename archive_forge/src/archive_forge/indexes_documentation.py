from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firestore import api_utils
Performs a Firestore Admin v1 Index list.

  Args:
    project: the project of the database of the index, a string.
    database: the database id of the index, a string.

  Returns:
    a list of Indexes.
  