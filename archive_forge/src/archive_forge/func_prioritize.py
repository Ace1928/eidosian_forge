import base64
from enum import Enum, IntEnum
from hyperframe.exceptions import InvalidPaddingError
from hyperframe.frame import (
from hpack.hpack import Encoder, Decoder
from hpack.exceptions import HPACKError, OversizedHeaderListError
from .config import H2Configuration
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .frame_buffer import FrameBuffer
from .settings import Settings, SettingCodes
from .stream import H2Stream, StreamClosedBy
from .utilities import SizeLimitDict, guard_increment_window
from .windows import WindowManager
def prioritize(self, stream_id, weight=None, depends_on=None, exclusive=None):
    """
        Notify a server about the priority of a stream.

        Stream priorities are a form of guidance to a remote server: they
        inform the server about how important a given response is, so that the
        server may allocate its resources (e.g. bandwidth, CPU time, etc.)
        accordingly. This exists to allow clients to ensure that the most
        important data arrives earlier, while less important data does not
        starve out the more important data.

        Stream priorities are explained in depth in `RFC 7540 Section 5.3
        <https://tools.ietf.org/html/rfc7540#section-5.3>`_.

        This method updates the priority information of a single stream. It may
        be called well before a stream is actively in use, or well after a
        stream is closed.

        .. warning:: RFC 7540 allows for servers to change the priority of
                     streams. However, hyper-h2 **does not** allow server
                     stacks to do this. This is because most clients do not
                     adequately know how to respond when provided conflicting
                     priority information, and relatively little utility is
                     provided by making that functionality available.

        .. note:: hyper-h2 **does not** maintain any information about the
                  RFC 7540 priority tree. That means that hyper-h2 does not
                  prevent incautious users from creating invalid priority
                  trees, particularly by creating priority loops. While some
                  basic error checking is provided by hyper-h2, users are
                  strongly recommended to understand their prioritisation
                  strategies before using the priority tools here.

        .. note:: Priority information is strictly advisory. Servers are
                  allowed to disregard it entirely. Avoid relying on the idea
                  that your priority signaling will definitely be obeyed.

        .. versionadded:: 2.4.0

        :param stream_id: The ID of the stream to prioritize.
        :type stream_id: ``int``

        :param weight: The weight to give the stream. Defaults to ``16``, the
             default weight of any stream. May be any value between ``1`` and
             ``256`` inclusive. The relative weight of a stream indicates what
             proportion of available resources will be allocated to that
             stream.
        :type weight: ``int``

        :param depends_on: The ID of the stream on which this stream depends.
             This stream will only be progressed if it is impossible to
             progress the parent stream (the one on which this one depends).
             Passing the value ``0`` means that this stream does not depend on
             any other. Defaults to ``0``.
        :type depends_on: ``int``

        :param exclusive: Whether this stream is an exclusive dependency of its
            "parent" stream (i.e. the stream given by ``depends_on``). If a
            stream is an exclusive dependency of another, that means that all
            previously-set children of the parent are moved to become children
            of the new exclusively-dependent stream. Defaults to ``False``.
        :type exclusive: ``bool``
        """
    if not self.config.client_side:
        raise RFC1122Error('Servers SHOULD NOT prioritize streams.')
    self.state_machine.process_input(ConnectionInputs.SEND_PRIORITY)
    frame = PriorityFrame(stream_id)
    frame = _add_frame_priority(frame, weight, depends_on, exclusive)
    self._prepare_for_sending([frame])