from gensim import corpora, models

# News articles
articles = [
    "The government announced a new economic stimulus package to boost the country's economy.",
    "A major technology company unveiled its latest smartphone model at a highly anticipated event.",
    "Scientists discovered a new species of dinosaur in a remote region of South America.",
    "The stock market experienced significant volatility amid concerns over trade tensions.",
    "A renowned artist opened a new exhibition showcasing their latest works.",
]

# Tokenize the articles
tokenized_articles = [article.lower().split() for article in articles]

# Create a dictionary and corpus
dictionary = corpora.Dictionary(tokenized_articles)
corpus = [dictionary.doc2bow(article) for article in tokenized_articles]

# Train an LDA model
lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=10)

# Print the discovered topics
for idx, topic in lda_model.print_topics():
    print(f"Topic {idx}: {topic}")
