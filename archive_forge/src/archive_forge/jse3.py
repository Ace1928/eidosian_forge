import os
import logging
from convokit import Corpus, TextParser, PolitenessStrategies, download
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define directories
RAW_DATA_DIR = "/home/lloyd/Downloads/indegodata/raw_datasets"
PROCESSED_DATA_DIR = "/home/lloyd/Downloads/indegodata/processed_data"

# List of datasets to download from ConvoKit
CONVOKIT_DATASETS = [
    "supreme-corpus",
    "wiki-corpus",
    "reddit-corpus-small",
    "chromium-corpus",
    "winning-args-corpus",
    "reddit-coarse-discourse-corpus",
    "persuasionforgood-corpus",
    "iq2-corpus",
    "friends-corpus",
    "switchboard-corpus",
    "wikipedia-politeness-corpus",
    "stack-exchange-politeness-corpus",
    "diplomacy-corpus",
    "gap-corpus",
    "casino-corpus",
]


# Function to download and process ConvoKit datasets
def process_convokit_datasets(datasets, raw_data_dir, processed_data_dir):
    corpora = {}
    for dataset in tqdm(datasets, desc="Processing ConvoKit datasets"):
        try:
            logging.info(f"Downloading dataset: {dataset}")
            corpus = Corpus(filename=download(dataset, data_dir=raw_data_dir))
            corpora[dataset] = corpus

            # Process all available configurations and splits
            for config in corpus.get_available_configs():
                for split in corpus.get_available_splits(config):
                    logging.info(f"Processing config: {config}, split: {split}")
                    split_corpus = corpus.load(config=config, split=split)
                    split_corpus.dump(
                        os.path.join(processed_data_dir, f"{dataset}_{config}_{split}")
                    )
        except Exception as e:
            logging.error(f"Failed to download dataset {dataset}: {e}")

    # Merging datasets into a universal corpus
    universal_corpus = corpora[list(corpora.keys())[0]]
    for dataset in list(corpora.keys())[1:]:
        universal_corpus = universal_corpus.merge(corpora[dataset])

    # Save the merged corpus
    universal_corpus.dump(os.path.join(processed_data_dir, "convokit_corpus"))

    return universal_corpus


# Function to apply transformers to a corpus
def apply_transformers(corpus):
    parser = TextParser()
    politeness = PolitenessStrategies()
    logging.info("Applying transformers to corpus")
    corpus = parser.transform(corpus)
    corpus = politeness.transform(corpus)
    return corpus


# Converting corpus components to DataFrames
def corpus_to_dataframes(corpus, prefix, processed_data_dir):
    try:
        utterances_df = corpus.get_utterances_dataframe()
        conversations_df = corpus.get_conversations_dataframe()
        speakers_df = corpus.get_speakers_dataframe()

        # Save the dataframes to CSV for further use
        utterances_df.to_csv(
            os.path.join(processed_data_dir, f"{prefix}_utterances.csv"), index=False
        )
        conversations_df.to_csv(
            os.path.join(processed_data_dir, f"{prefix}_conversations.csv"), index=False
        )
        speakers_df.to_csv(
            os.path.join(processed_data_dir, f"{prefix}_speakers.csv"), index=False
        )

        return utterances_df, conversations_df, speakers_df
    except Exception as e:
        logging.error(f"Failed to convert corpus to DataFrames: {e}")
        raise


# ConvoKit Processor
def convokit_processor():
    convokit_corpus = process_convokit_datasets(
        CONVOKIT_DATASETS, RAW_DATA_DIR, PROCESSED_DATA_DIR
    )
    convokit_corpus = apply_transformers(convokit_corpus)
    convokit_utterances_df, convokit_conversations_df, convokit_speakers_df = (
        corpus_to_dataframes(convokit_corpus, "convokit", PROCESSED_DATA_DIR)
    )
    logging.info("Displaying the first few rows of the ConvoKit utterances DataFrame")
    print(convokit_utterances_df.head())


# Execute processor
convokit_processor()
