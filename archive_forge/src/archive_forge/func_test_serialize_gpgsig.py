import datetime
import os
import stat
from contextlib import contextmanager
from io import BytesIO
from itertools import permutations
from dulwich.tests import TestCase
from ..errors import ObjectFormatException
from ..objects import (
from .utils import ext_functest_builder, functest_builder, make_commit, make_object
def test_serialize_gpgsig(self):
    commit = self.make_commit(gpgsig=b'-----BEGIN PGP SIGNATURE-----\nVersion: GnuPG v1\n\niQIcBAABCgAGBQJULCdfAAoJEACAbyvXKaRXuKwP/RyP9PA49uAvu8tQVCC/uBa8\nvi975+xvO14R8Pp8k2nps7lSxCdtCd+xVT1VRHs0wNhOZo2YCVoU1HATkPejqSeV\nNScTHcxnk4/+bxyfk14xvJkNp7FlQ3npmBkA+lbV0Ubr33rvtIE5jiJPyz+SgWAg\nxdBG2TojV0squj00GoH/euK6aX7GgZtwdtpTv44haCQdSuPGDcI4TORqR6YSqvy3\nGPE+3ZqXPFFb+KILtimkxitdwB7CpwmNse2vE3rONSwTvi8nq3ZoQYNY73CQGkUy\nqoFU0pDtw87U3niFin1ZccDgH0bB6624sLViqrjcbYJeg815Htsu4rmzVaZADEVC\nXhIO4MThebusdk0AcNGjgpf3HRHk0DPMDDlIjm+Oao0cqovvF6VyYmcb0C+RmhJj\ndodLXMNmbqErwTk3zEkW0yZvNIYXH7m9SokPCZa4eeIM7be62X6h1mbt0/IU6Th+\nv18fS0iTMP/Viug5und+05C/v04kgDo0CPphAbXwWMnkE4B6Tl9sdyUYXtvQsL7x\n0+WP1gL27ANqNZiI07Kz/BhbBAQI/+2TFT7oGr0AnFPQ5jHp+3GpUf6OKuT1wT3H\nND189UFuRuubxb42vZhpcXRbqJVWnbECTKVUPsGZqat3enQUB63uM4i6/RdONDZA\nfDeF1m4qYs+cUXKNUZ03\n=X6RT\n-----END PGP SIGNATURE-----')
    self.maxDiff = None
    self.assertEqual(b'tree d80c186a03f423a81b39df39dc87fd269736ca86\nparent ab64bbdcc51b170d21588e5c5d391ee5c0c96dfd\nparent 4cffe90e0a41ad3f5190079d7c8f036bde29cbe6\nauthor James Westby <jw+debian@jameswestby.net> 1174773719 +0000\ncommitter James Westby <jw+debian@jameswestby.net> 1174773719 +0000\ngpgsig -----BEGIN PGP SIGNATURE-----\n Version: GnuPG v1\n \n iQIcBAABCgAGBQJULCdfAAoJEACAbyvXKaRXuKwP/RyP9PA49uAvu8tQVCC/uBa8\n vi975+xvO14R8Pp8k2nps7lSxCdtCd+xVT1VRHs0wNhOZo2YCVoU1HATkPejqSeV\n NScTHcxnk4/+bxyfk14xvJkNp7FlQ3npmBkA+lbV0Ubr33rvtIE5jiJPyz+SgWAg\n xdBG2TojV0squj00GoH/euK6aX7GgZtwdtpTv44haCQdSuPGDcI4TORqR6YSqvy3\n GPE+3ZqXPFFb+KILtimkxitdwB7CpwmNse2vE3rONSwTvi8nq3ZoQYNY73CQGkUy\n qoFU0pDtw87U3niFin1ZccDgH0bB6624sLViqrjcbYJeg815Htsu4rmzVaZADEVC\n XhIO4MThebusdk0AcNGjgpf3HRHk0DPMDDlIjm+Oao0cqovvF6VyYmcb0C+RmhJj\n dodLXMNmbqErwTk3zEkW0yZvNIYXH7m9SokPCZa4eeIM7be62X6h1mbt0/IU6Th+\n v18fS0iTMP/Viug5und+05C/v04kgDo0CPphAbXwWMnkE4B6Tl9sdyUYXtvQsL7x\n 0+WP1gL27ANqNZiI07Kz/BhbBAQI/+2TFT7oGr0AnFPQ5jHp+3GpUf6OKuT1wT3H\n ND189UFuRuubxb42vZhpcXRbqJVWnbECTKVUPsGZqat3enQUB63uM4i6/RdONDZA\n fDeF1m4qYs+cUXKNUZ03\n =X6RT\n -----END PGP SIGNATURE-----\n\nMerge ../b\n', commit.as_raw_string())