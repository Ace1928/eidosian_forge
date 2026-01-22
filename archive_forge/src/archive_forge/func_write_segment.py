from .segmented import SegmentedStream
def write_segment(self, data):
    """
        :param data:
        :type data: bytes
        :return:
        :rtype:
        """
    self._sent.append(data)