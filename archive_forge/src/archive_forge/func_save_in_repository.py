from collections import defaultdict
from io import BytesIO
from .. import errors, trace
from .. import transport as _mod_transport
def save_in_repository(self, repository):
    with BytesIO() as f:
        self.save(f)
        f.seek(0)
        repository.control_transport.put_file('git-unpeel-map', f)