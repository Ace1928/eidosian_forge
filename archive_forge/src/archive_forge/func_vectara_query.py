from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass, field
from hashlib import md5
from typing import Any, Iterable, List, Optional, Tuple, Type
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Field
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
def vectara_query(self, query: str, config: VectaraQueryConfig, **kwargs: Any) -> List[Tuple[Document, float]]:
    """Run a Vectara query

        Args:
            query: Text to look up documents similar to.
            config: VectaraQueryConfig object
        Returns:
            A list of k Documents matching the given query
            If summary is enabled, last document is the summary text with 'summary'=True
        """
    if isinstance(config.mmr_config, dict):
        config.mmr_config = MMRConfig(**config.mmr_config)
    if isinstance(config.summary_config, dict):
        config.summary_config = SummaryConfig(**config.summary_config)
    data = {'query': [{'query': query, 'start': 0, 'numResults': config.mmr_config.mmr_k if config.mmr_config.is_enabled else config.k, 'contextConfig': {'sentencesBefore': config.n_sentence_context, 'sentencesAfter': config.n_sentence_context}, 'corpusKey': [{'customerId': self._vectara_customer_id, 'corpusId': self._vectara_corpus_id, 'metadataFilter': config.filter, 'lexicalInterpolationConfig': {'lambda': config.lambda_val}}]}]}
    if config.mmr_config.is_enabled:
        data['query'][0]['rerankingConfig'] = {'rerankerId': 272725718, 'mmrConfig': {'diversityBias': config.mmr_config.diversity_bias}}
    if config.summary_config.is_enabled:
        data['query'][0]['summary'] = [{'maxSummarizedResults': config.summary_config.max_results, 'responseLang': config.summary_config.response_lang, 'summarizerPromptName': config.summary_config.prompt_name}]
    response = self._session.post(headers=self._get_post_headers(), url='https://api.vectara.io/v1/query', data=json.dumps(data), timeout=self.vectara_api_timeout)
    if response.status_code != 200:
        logger.error('Query failed %s', f'(code {response.status_code}, reason {response.reason}, details {response.text})')
        return ([], '')
    result = response.json()
    if config.score_threshold:
        responses = [r for r in result['responseSet'][0]['response'] if r['score'] > config.score_threshold]
    else:
        responses = result['responseSet'][0]['response']
    documents = result['responseSet'][0]['document']
    metadatas = []
    for x in responses:
        md = {m['name']: m['value'] for m in x['metadata']}
        doc_num = x['documentIndex']
        doc_md = {m['name']: m['value'] for m in documents[doc_num]['metadata']}
        if 'source' not in doc_md:
            doc_md['source'] = 'vectara'
        md.update(doc_md)
        metadatas.append(md)
    res = [(Document(page_content=x['text'], metadata=md), x['score']) for x, md in zip(responses, metadatas)]
    if config.mmr_config.is_enabled:
        res = res[:config.k]
    if config.summary_config.is_enabled:
        summary = result['responseSet'][0]['summary'][0]['text']
        res.append((Document(page_content=summary, metadata={'summary': True}), 0.0))
    return res