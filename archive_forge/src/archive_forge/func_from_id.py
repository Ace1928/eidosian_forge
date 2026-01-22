from __future__ import unicode_literals
@classmethod
def from_id(cls, qid):
    if qid in cls._id_map:
        return cls._id_map[qid]
    raise ValueError('{} is not a valid {} value'.format(qid, cls.__name__))