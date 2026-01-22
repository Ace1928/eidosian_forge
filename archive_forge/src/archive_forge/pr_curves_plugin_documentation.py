import numpy as np
from werkzeug import wrappers
from tensorboard import plugin_util
from tensorboard.data import provider
from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin
from tensorboard.plugins.pr_curve import metadata
Creates an entry for PR curve data. Each entry corresponds to 1
        step.

        Args:
          step: The step.
          wall_time: The wall time.
          data_array: A numpy array of PR curve data stored in the summary format.

        Returns:
          A PR curve entry.
        