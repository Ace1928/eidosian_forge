import itertools
from gzip import GzipFile
from io import BytesIO
from ... import errors
from ... import graph as _mod_graph
from ... import osutils, progress, transport, ui
from ...errors import RevisionAlreadyPresent, RevisionNotPresent
from ...tests import (TestCase, TestCaseWithMemoryTransport, TestNotApplicable,
from ...tests.http_utils import TestCaseWithWebserver
from ...tests.scenarios import load_tests_apply_scenarios
from ...transport.memory import MemoryTransport
from .. import groupcompress
from .. import knit as _mod_knit
from .. import versionedfile as versionedfile
from ..knit import cleanup_pack_knit, make_file_factory, make_pack_factory
from ..versionedfile import (ChunkedContentFactory, ConstantMapper,
from ..weave import WeaveFile, WeaveInvalidChecksum
from ..weavefile import write_weave
def test_deannotation_noeol(self):
    """Test converting annotated knits to unannotated knits."""
    f = self.get_knit()
    get_diamond_files(f, 1, trailing_eol=False)
    ft_data, delta_data = self.helpGetBytes(f, 'knit-ft-gz', _mod_knit.FTAnnotatedToUnannotated(None), 'knit-delta-gz', _mod_knit.DeltaAnnotatedToUnannotated(None))
    self.assertEqual(b'version origin 1 b284f94827db1fa2970d9e2014f080413b547a7e\norigin\nend origin\n', GzipFile(mode='rb', fileobj=BytesIO(ft_data)).read())
    self.assertEqual(b'version merged 4 32c2e79763b3f90e8ccde37f9710b6629c25a796\n1,2,3\nleft\nright\nmerged\nend merged\n', GzipFile(mode='rb', fileobj=BytesIO(delta_data)).read())