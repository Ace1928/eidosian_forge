from ._base import *
from .filters import GDELTFilters
from .models import GDELTArticle
def timeline_search(self, mode: str, filters: GDELTFilters) -> Union[pd.DataFrame, Dict, str]:
    timeline = self._query(mode, filters.query_string)
    results = {'datetime': [entry['date'] for entry in timeline['timeline'][0]['data']]}
    for series in timeline['timeline']:
        results[series['series']] = [entry['value'] for entry in series['data']]
    if mode == 'timelinevolraw':
        results['All Articles'] = [entry['norm'] for entry in timeline['timeline'][0]['data']]
    return self.return_timeline_search(results)