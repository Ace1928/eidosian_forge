import logging
import pkg_resources
from tensorboard.plugins.audio import audio_plugin
from tensorboard.plugins.core import core_plugin
from tensorboard.plugins.custom_scalar import custom_scalars_plugin
from tensorboard.plugins.debugger_v2 import debugger_v2_plugin
from tensorboard.plugins.distribution import distributions_plugin
from tensorboard.plugins.graph import graphs_plugin
from tensorboard.plugins.histogram import histograms_plugin
from tensorboard.plugins.hparams import hparams_plugin
from tensorboard.plugins.image import images_plugin
from tensorboard.plugins.metrics import metrics_plugin
from tensorboard.plugins.pr_curve import pr_curves_plugin
from tensorboard.plugins.profile_redirect import profile_redirect_plugin
from tensorboard.plugins.scalar import scalars_plugin
from tensorboard.plugins.text import text_plugin
from tensorboard.plugins.mesh import mesh_plugin
from tensorboard.plugins.wit_redirect import wit_redirect_plugin
def get_static_plugins():
    """Returns a list specifying TensorBoard's default first-party plugins.

    Plugins are specified in this list either via a TBLoader instance to load the
    plugin, or the TBPlugin class itself which will be loaded using a BasicLoader.

    This list can be passed to the `tensorboard.program.TensorBoard` API.

    Returns:
      The list of default first-party plugins.

    :rtype: list[Type[base_plugin.TBLoader] | Type[base_plugin.TBPlugin]]
    """
    return _PLUGINS[:]