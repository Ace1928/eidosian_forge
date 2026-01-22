from gitdb.db.loose import LooseObjectDB
from gitdb.db.base import (
from gitdb.base import (
from gitdb.exc import (
from gitdb.stream import (
from io import BytesIO
Copy the streams as identified by sha's yielded by sha_iter into the given odb
        The streams will be copied directly
        **Note:** the object will only be written if it did not exist in the target db

        :return: amount of streams actually copied into odb. If smaller than the amount
            of input shas, one or more objects did already exist in odb