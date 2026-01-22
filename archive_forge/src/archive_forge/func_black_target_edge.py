def black_target_edge(self, e):
    """
        This is invoked on the subset of non-tree edges whose target vertex is
        colored black at the time of examination.
        The color black indicates that the vertex has been removed from the queue.
        """
    return