import re
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
def promote_subtitle(self, node):
    """
        Transform the following node tree::

            <node>
                <title>
                <section>
                    <title>
                    ...

        into ::

            <node>
                <title>
                <subtitle>
                ...
        """
    if not isinstance(node, nodes.Element):
        raise TypeError('node must be of Element-derived type.')
    subsection, index = self.candidate_index(node)
    if index is None:
        return None
    subtitle = nodes.subtitle()
    subtitle.update_all_atts_concatenating(subsection, True, True)
    subtitle[:] = subsection[0][:]
    node[:] = node[:1] + [subtitle] + node[1:index] + subsection[1:]
    return 1