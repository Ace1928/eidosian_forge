import sqlalchemy as sa
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import Table
from ._autogen_fixtures import AutogenFixtureTest
from ... import testing
from ...testing import config
from ...testing import eq_
from ...testing import exclusions
from ...testing import is_
from ...testing import is_true
from ...testing import mock
from ...testing import TestBase
Asserts the current behavior which is that on PG and Oracle,
        the GENERATED ALWAYS AS is reflected as a server default which we can't
        tell is actually "computed", so these come out as a modification to
        the server default.

        