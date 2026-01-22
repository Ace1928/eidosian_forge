from information_schema.collations order by id;" | python -c "import sys
def by_id(self, id):
    return self._by_id[id]