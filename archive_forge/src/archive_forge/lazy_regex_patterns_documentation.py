from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
A list of regex that can be compiled lazily to improve gcloud performance.

Some regex need to be compiled immediately so that they will raise an
re.error if the pattern is not valid, but the below list are known to be
valid, so they can be compiled lazily. Most of these regex don't end up being
compiled for running any given gcloud command.

These patterns are kept in a Python source file to minimize loading time.

They should be updated periodically with //cloud/sdk:update_lazy_regex.
