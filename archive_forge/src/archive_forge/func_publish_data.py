import warnings
from traitlets import Any, CBytes, Dict, Instance
from traitlets.config import Configurable
from ipykernel.jsonutil import json_clean
from jupyter_client.session import Session, extract_header
def publish_data(self, data):
    """publish a data_message on the IOPub channel

        Parameters
        ----------
        data : dict
            The data to be published. Think of it as a namespace.
        """
    session = self.session
    assert session is not None
    buffers = serialize_object(data, buffer_threshold=session.buffer_threshold, item_threshold=session.item_threshold)
    content = json_clean(dict(keys=list(data.keys())))
    session.send(self.pub_socket, 'data_message', content=content, parent=self.parent_header, buffers=buffers, ident=self.topic)