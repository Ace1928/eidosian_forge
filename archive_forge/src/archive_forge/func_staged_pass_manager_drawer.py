from __future__ import annotations
import os
import inspect
import tempfile
from qiskit.utils import optionals as _optionals
from qiskit.passmanager.base_tasks import BaseController, GenericPass
from qiskit.passmanager.flow_controllers import FlowControllerLinear
from qiskit.transpiler.basepasses import AnalysisPass, TransformationPass
from .exceptions import VisualizationError
@_optionals.HAS_GRAPHVIZ.require_in_call
@_optionals.HAS_PYDOT.require_in_call
def staged_pass_manager_drawer(pass_manager, filename=None, style=None, raw=False):
    """
    Draws the staged pass manager.

        This function needs `pydot <https://github.com/erocarrera/pydot>`__, which in turn needs
    `Graphviz <https://www.graphviz.org/>`__ to be installed.

    Args:
        pass_manager (StagedPassManager): the staged pass manager to be drawn
        filename (str): file path to save image to
        style (dict or OrderedDict): keys are the pass classes and the values are
            the colors to make them. An example can be seen in the DEFAULT_STYLE. An ordered
            dict can be used to ensure a priority coloring when pass falls into multiple
            categories. Any values not included in the provided dict will be filled in from
            the default dict
        raw (Bool) : True if you want to save the raw Dot output not an image. The
            default is False.
    Returns:
        PIL.Image or None: an in-memory representation of the pass manager. Or None if
        no image was generated or PIL is not installed.
    Raises:
        MissingOptionalLibraryError: when nxpd or pydot not installed.
        VisualizationError: If raw=True and filename=None.

    Example:
        .. code-block::

            %matplotlib inline
            from qiskit.providers.fake_provider import GenericBackendV2
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

            pass_manager = generate_preset_pass_manager(3, GenericBackendV2(num_qubits=5))
            pass_manager.draw()
    """
    import pydot
    stages = list(filter(lambda s: s is not None, pass_manager.expanded_stages))
    if not style:
        style = DEFAULT_STYLE
    graph = pydot.Dot()
    component_id = 0
    idx = 0
    prev_node = None
    for st in stages:
        stage = getattr(pass_manager, st)
        if stage is not None:
            stagegraph = pydot.Cluster(str(st), label=str(st), fontname='helvetica', labeljust='l')
            for controller_group in stage.to_flow_controller().tasks:
                subgraph, component_id, prev_node = draw_subgraph(controller_group, component_id, style, prev_node, idx)
                stagegraph.add_subgraph(subgraph)
                idx += 1
            graph.add_subgraph(stagegraph)
    output = make_output(graph, raw, filename)
    return output