import abc
import datetime
from dogpile.cache import api
from dogpile import util as dp_util
from oslo_cache import core
from oslo_log import log
from oslo_utils import importutils
from oslo_utils import timeutils
from oslo_cache._i18n import _
from oslo_cache import exception
def get_cache_collection(self):
    if self.cache_collection not in self._MONGO_COLLS:
        global pymongo
        import pymongo
        if self.db_name not in self._DB:
            self._DB[self.db_name] = self._get_db()
        coll = getattr(self._DB[self.db_name], self.cache_collection)
        self._assign_data_mainpulator()
        if self.read_preference:
            f = getattr(pymongo.read_preferences, 'read_pref_mode_from_name', None)
            if not f:
                f = pymongo.read_preferences.mongos_enum
            self.read_preference = f(self.read_preference)
            coll.read_preference = self.read_preference
        if self.w > -1:
            coll.write_concern['w'] = self.w
        if self.ttl_seconds > 0:
            kwargs = {'expireAfterSeconds': self.ttl_seconds}
            coll.ensure_index('doc_date', cache_for=5, **kwargs)
        else:
            self._validate_ttl_index(coll, self.cache_collection, self.ttl_seconds)
        self._MONGO_COLLS[self.cache_collection] = coll
    return self._MONGO_COLLS[self.cache_collection]