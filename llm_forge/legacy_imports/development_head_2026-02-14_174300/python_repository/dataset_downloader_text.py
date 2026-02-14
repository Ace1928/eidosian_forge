import os
import json
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
import shutil
from typing import List, Dict, Union, Optional
import glob
import random
import docx
import PyPDF2
import chardet
import logging
import concurrent.futures
import colorama
from colorama import Fore, Style
import statistics
import re


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

# Initialize colorama
colorama.init(autoreset=True)

def _detect_encoding(file_path: str) -> str:
    """
    Detects the encoding of a file.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: Detected file encoding.
    """
    with open(file_path, 'rb') as f:
        raw = f.read()
        result = chardet.detect(raw)
        return result['encoding']

def _process_file(file_path: str) -> str:
    """
    Processes a single file into a standardized single-line format.

    Args:
        file_path (str): Path to file to process.

    Returns:
        str: Processed text in single-line format.

    Handles multiple file types:
      • .txt, .py: Preserves newlines as \n
      • .json/.jsonl: Preserves overall JSON structure, converting them to a single line
      • .docx: Extracts text from Word documents
      • .pdf: Extracts text from PDF pages
    """
    try:
        ext = os.path.splitext(file_path)[1].lower()
        logger.debug(f"Processing file: {file_path}, detected extension: {ext}")

        if ext == '.txt' or ext == '.py':
            encoding = _detect_encoding(file_path)
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read().replace('\n', '\\n')

        elif ext == '.json' or ext == '.jsonl':
            # Load JSON or JSONL into "content"
            with open(file_path, 'r', encoding='utf-8') as f:
                content = (
                    json.load(f)
                    if ext == '.json'
                    else [json.loads(line) for line in f]
                )
            return json.dumps(content).replace('\n', '\\n')

        elif ext == '.docx':
            doc_file = docx.Document(file_path)
            return '\\n'.join(paragraph.text for paragraph in doc_file.paragraphs)

        elif ext == '.pdf':
            text_content = []
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text_content.append(page.extract_text() or "")
            return '\\n'.join(text_content)

        else:
            logger.warning(f"Unrecognized file extension for {file_path}. Returning empty string.")
            return ""
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return ""

def _save_dataset_to_jsonl(dataset, jsonl_path: str) -> None:
    """
    Saves a Hugging Face dataset to a JSON Lines file, using concurrency for improved performance.

    Args:
        dataset: The Hugging Face dataset to save.
        jsonl_path (str): The path to save the JSON Lines file.
    """
    logger.debug(f"Saving dataset to {jsonl_path}")
    items = list(dataset)  # Convert the dataset to a list for parallel processing

    # Convert each dataset item to JSON lines in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        json_lines = list(executor.map(json.dumps, items))

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for line in tqdm(json_lines, desc=f"Writing {os.path.basename(jsonl_path)}"):
            f.write(line + "\n")

    logger.debug(f"Finished writing dataset to {jsonl_path}")

class DatasetHandler:
    """
    A comprehensive class for downloading, processing, and preparing text datasets
    for language model training and code generation tasks.

    This class provides a unified interface for:
    - Downloading datasets from various sources (HuggingFace, local files)
    - Processing multiple file formats (txt, py, json, docx, pdf) into a standardized format
    - Saving datasets in consistent formats (JSONL, TXT)
    - Loading and preprocessing datasets for LLM training
    - Handling dataset splits (train/validation/test)
    - Converting all documents to single-line format while preserving structure

    Supported dataset sources:
      • HuggingFace datasets (text8, openai_humaneval, mbpp, tfix, helpsteer2)
      • Local files (.txt, .py, .json, .jsonl, .docx, .pdf)
      • Directories of documents (auto-split into train/val/test)

    All text is processed into a universal single-line format suitable for modern LLMs,
    preserving newlines as '\n' and document structure. This class includes advanced logging,
    optional caching to skip already-downloaded datasets, and concurrency to speed up saving.
    """

    def __init__(self, data_dir: str = "./datasets", enable_cache: bool = True):
        """
        Initializes the DatasetHandler with enhanced logging, optional caching, and concurrency.

        Args:
            data_dir (str): Base directory for storing all datasets and processed files.
                            Creates directory if it doesn't exist.
            enable_cache (bool): If True, checks whether a dataset folder already exists to skip re-download.
        """
        self.data_dir = data_dir
        self.enable_cache = enable_cache
        os.makedirs(self.data_dir, exist_ok=True)

    def download_and_save_dataset(self, dataset_name: str, split_ratios: Optional[Dict[str, float]] = None) -> None:
        """
        Downloads a dataset, saves it to disk, and optionally splits it.

        Args:
            dataset_name (str): The name of the dataset to download.
            split_ratios (dict, optional): Ratios for train/validation/test splits (e.g. {'train':0.8, 'validation':0.1, 'test':0.1}).
                                           Defaults to None (no splitting).
        """
        logger.info(f"Requested download of dataset: {dataset_name}")
        dataset_path = os.path.join(self.data_dir, dataset_name)
        os.makedirs(dataset_path, exist_ok=True)

        # If caching is enabled and directory is already populated, skip re-download
        if self.enable_cache and os.listdir(dataset_path):
            logger.info(f"Dataset directory '{dataset_path}' is not empty. Skipping download due to caching.")
            print(f"Skipping download of {dataset_name} (cached).")
            return

        print(f"Downloading and preparing dataset: {dataset_name}...")
        logger.debug(f"Created/confirmed dataset directory at {dataset_path}")

        if dataset_name == "text8":
            self._prepare_text8(dataset_path)
        elif dataset_name == "openai_humaneval":
            self._prepare_humaneval(dataset_path)
        elif dataset_name == "mbpp":
            self._prepare_mbpp(dataset_path)
        elif dataset_name == "tfix":
            self._prepare_tfix(dataset_path)
        elif dataset_name == "helpsteer2":
            self._prepare_helpsteer2(dataset_path, split_ratios)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        logger.info(f"Dataset {dataset_name} downloaded and saved to {dataset_path}")
        print(f"Dataset {dataset_name} downloaded and saved to {dataset_path}")

    def _prepare_text8(self, dataset_path: str) -> None:
        """
        Downloads and prepares the Text8 dataset.

        Args:
            dataset_path (str): The path where the dataset will be saved.
        """
        dataset = load_dataset("afmck/text8")["train"]
        text = " ".join(dataset["text"])
        with open(os.path.join(dataset_path, "text8.txt"), "w", encoding="utf-8") as f:
            f.write(text)

    def _prepare_humaneval(self, dataset_path: str) -> None:
        """
        Downloads and prepares the HumanEval dataset.

        Args:
            dataset_path (str): The path where the dataset will be saved.
        """
        dataset = load_dataset("openai_humaneval")["test"]
        _save_dataset_to_jsonl(dataset, os.path.join(dataset_path, "humaneval.jsonl"))

    def _prepare_mbpp(self, dataset_path: str) -> None:
        """
        Downloads and prepares the MBPP dataset.

        Args:
            dataset_path (str): The path where the dataset will be saved.
        """
        dataset = load_dataset("mbpp")["train"]
        _save_dataset_to_jsonl(dataset, os.path.join(dataset_path, "mbpp.jsonl"))

    def _prepare_tfix(self, dataset_path: str) -> None:
        """
        Downloads and prepares the Tfix dataset.

        Args:
            dataset_path (str): The path where the dataset will be saved.
        """
        dataset = load_dataset("bigcode/tfix")['train']
        dataset = dataset.train_test_split(test_size=0.1)

        _save_dataset_to_jsonl(dataset['train'], os.path.join(dataset_path, "train.jsonl"))
        _save_dataset_to_jsonl(dataset['test'], os.path.join(dataset_path, "test.jsonl"))

    def _prepare_helpsteer2(self, dataset_path: str, split_ratios: Optional[Dict[str, float]]) -> None:
        """
        Downloads and prepares the HelpSteer2 dataset.

        Args:
            dataset_path (str): The path where the dataset will be saved.
            split_ratios (dict): Ratios for train/validation/test splits.
        """
        dataset = load_dataset("nvidia/helpsteer2")['train']

        if split_ratios:
            dataset = dataset.train_test_split(
                test_size=(split_ratios['test'] + split_ratios['validation'])
            )
            train_dataset = dataset['train']
            temp_dataset = dataset['test'].train_test_split(
                test_size=(split_ratios['test'] / (split_ratios['test'] + split_ratios['validation']))
            )
            validation_dataset = temp_dataset['train']
            test_dataset = temp_dataset['test']

            _save_dataset_to_jsonl(train_dataset, os.path.join(dataset_path, "train.jsonl"))
            _save_dataset_to_jsonl(validation_dataset, os.path.join(dataset_path, "validation.jsonl"))
            _save_dataset_to_jsonl(test_dataset, os.path.join(dataset_path, "test.jsonl"))
        else:
            _save_dataset_to_jsonl(dataset, os.path.join(dataset_path, "helpsteer2.jsonl"))

    def process_directory(self, dir_path: str, split_ratios: Dict[str, float] = {'train': 0.8, 'validation': 0.1, 'test': 0.1}) -> Dict[str, List[str]]:
        """
        Processes all supported files in a directory into train/val/test splits.

        Args:
            dir_path (str): Directory containing files to process.
            split_ratios (dict): Ratios for train/validation/test splits.

        Returns:
            dict: Split datasets as {'train': [...], 'validation': [...], 'test': [...]}.
        """
        supported_extensions = ['.txt', '.py', '.json', '.jsonl', '.docx', '.pdf']
        all_files = []
        for ext in supported_extensions:
            all_files.extend(glob.glob(os.path.join(dir_path, f"*{ext}")))

        random.shuffle(all_files)

        train_idx = int(len(all_files) * split_ratios['train'])
        val_idx = train_idx + int(len(all_files) * split_ratios['validation'])

        splits = {
            'train': [_process_file(f) for f in all_files[:train_idx]],
            'validation': [_process_file(f) for f in all_files[train_idx:val_idx]],
            'test': [_process_file(f) for f in all_files[val_idx:]]
        }
        return splits

    def load_dataset_from_jsonl(self, dataset_name: str, split_name: str = "") -> List[Dict]:
        """
        Loads a dataset from a JSON Lines file.

        Args:
            dataset_name (str): The name of the dataset.
            split_name (str, optional): If split, one of train, test, or validation. 
                                        Otherwise returns the entire dataset.

        Returns:
            list: A list of dictionaries, where each dictionary represents a sample.
        """
        dataset_path = os.path.join(self.data_dir, dataset_name)
        if split_name == "":
            jsonl_path = os.path.join(dataset_path, f"{dataset_name}.jsonl")
        else:
            jsonl_path = os.path.join(dataset_path, f"{split_name}.jsonl")

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")

        data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def load_text_data(self, dataset_name: str, split_name: str = "") -> List[str]:
            """
            Universal loader for text data from any supported source.
            
            Provides preprocessed, ready-to-use text data for LLM training.
            Handles both HuggingFace datasets and local files/directories.
            
            Args:
                dataset_name (str): Name of dataset or path to file/directory
                split_name (str): Split to load (train/validation/test)
                                Empty string loads full dataset for single files
                                
            Returns:
                list: Lines of text ready for LLM training
                
            Raises:
                ValueError: If dataset/file format not supported
                FileNotFoundError: If dataset/file not found
            """
            if os.path.exists(dataset_name):
                if os.path.isfile(dataset_name):
                    return [_process_file(dataset_name)]
                elif os.path.isdir(dataset_name):
                    splits = self.process_directory(dataset_name)
                    return splits.get(split_name, splits['train'])
            
            dataset_path = os.path.join(self.data_dir, dataset_name)
            
            if dataset_name == "text8":
                with open(os.path.join(dataset_path, "text8.txt"), "r", encoding="utf-8") as f:
                    return [f.read().replace('\n', '\\n')]
            
            data = self.load_dataset_from_jsonl(dataset_name, split_name)
            
            if dataset_name == "helpsteer2":
                return [item["conversation"].replace('\n', '\\n') for item in data]
            elif dataset_name == "openai_humaneval":
                return [f"{item['prompt']}\\n{item['canonical_solution']}".replace('\n', '\\n') for item in data]
            elif dataset_name == "mbpp":
                return [f"{item['text']}\\n{item['code']}".replace('\n', '\\n') for item in data]
            elif dataset_name == "tfix":
                return [f"{item['bug']}\\n{item['fix']}".replace('\n', '\\n') for item in data]
            else:
                raise ValueError(f"Unsupported dataset format: {dataset_name}")

    def sample_and_display_entries(self, dataset_name: str, split_name: str = "", num_samples: int = 5) -> None:
        """
        Randomly samples a few entries from a loaded dataset and prints them to the console with colorization and 
        formatting for easy viewing.

        Args:
            dataset_name (str): Name/path of the dataset or file/directory to load.
            split_name (str): The specific split (train/validation/test). 
                              If empty, loads either single file or entire dataset.
            num_samples (int): Number of samples to display.
        """
        logger.info(f"Sampling up to {num_samples} entries from dataset '{dataset_name}' (split='{split_name}')")

        # Use universal loader for text data
        data_lines = self.load_text_data(dataset_name, split_name)
        if not data_lines:
            print(Fore.RED + "[No data lines available to sample.]" + Style.RESET_ALL)
            return

        samples = random.sample(data_lines, min(num_samples, len(data_lines)))
        print(Fore.GREEN + f"--- Showing {len(samples)} random samples from '{dataset_name}' (split='{split_name}') ---" + Style.RESET_ALL)
        for idx, line in enumerate(samples, start=1):
            # Print sample lines with a special color or formatting symbols
            print(Fore.CYAN + f"[Sample {idx}] " + Style.RESET_ALL + line)

        print(Fore.GREEN + "--- End of samples ---" + Style.RESET_ALL)


    def analyze_text_lines(self, lines: List[str], top_n: int = 10) -> Dict[str, Union[int, float, List[str]]]:
        """
        Computes basic statistics about a list of text lines, including:
          - Count of lines
          - Min, max, and average line length
          - Top 'n' frequent tokens (approximation)

        Args:
            lines (List[str]): The list of text lines (strings) to analyze.
            top_n (int): The number of top frequent tokens to return.

        Returns:
            dict: A dictionary containing analysis metrics.
        """
        analysis = {
            "count": len(lines),
            "min_length": 0,
            "max_length": 0,
            "avg_length": 0.0,
            "top_tokens": []
        }

        if not lines:
            return analysis

        # Measure line lengths
        lengths = [len(line) for line in lines]
        analysis["min_length"] = min(lengths)
        analysis["max_length"] = max(lengths)
        analysis["avg_length"] = statistics.mean(lengths)

        # Simple tokenization by non-alphabetic or non-numeric delimiters
        token_counts = {}
        token_pattern = re.compile(r"\w+")
        for line in lines:
            for token in token_pattern.findall(line.lower()):
                token_counts[token] = token_counts.get(token, 0) + 1

        # Sort by frequency descending and take top_n
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        analysis["top_tokens"] = sorted_tokens[:top_n]

        return analysis

    def perform_exploratory_analysis(self, dataset_name: str, split_name: str = "", num_samples: int = 5, top_n: int = 10) -> None:
        """
        Loads text data, prints random samples, and computes basic exploratory statistics.

        Args:
            dataset_name (str): Name/path of the dataset or file/directory to load.
            split_name (str): The specific split (train/validation/test). 
            num_samples (int): Number of sample lines to display.
            top_n (int): Number of top tokens to show in analysis.
        """
        # Step 1: Sample display
        self.sample_and_display_entries(dataset_name, split_name, num_samples)

        # Step 2: Basic stats
        lines = self.load_text_data(dataset_name, split_name)
        stats = self.analyze_text_lines(lines, top_n=top_n)

        # Step 3: Print stats colorfully
        print(Fore.MAGENTA + "\n--- Exploratory Analysis ---" + Style.RESET_ALL)
        print(f"Total lines: {stats['count']}")
        print(f"Min length: {stats['min_length']}")
        print(f"Max length: {stats['max_length']}")
        print(f"Average length: {stats['avg_length']:.2f}")
        print(Fore.YELLOW + f"\nTop {top_n} tokens:" + Style.RESET_ALL)
        for token, freq in stats["top_tokens"]:
            print(f"   {token}: {freq}")
        print(Fore.MAGENTA + "--- End of Analysis ---\n" + Style.RESET_ALL)


# === Add top-level convenience functions so they can be imported directly ===

def download_and_save_dataset(
    dataset_name: str, 
    split_ratios: Optional[Dict[str, float]] = None,
    data_dir: str = "./datasets", 
    enable_cache: bool = True
) -> None:
    """
    Convenience function to download and save a dataset without instantiating DatasetHandler.
    """
    handler = DatasetHandler(data_dir=data_dir, enable_cache=enable_cache)
    handler.download_and_save_dataset(dataset_name, split_ratios)

def process_directory(
    dir_path: str,
    split_ratios: Dict[str, float] = {'train': 0.8, 'validation': 0.1, 'test': 0.1},
    data_dir: str = "./datasets",
    enable_cache: bool = True
) -> Dict[str, List[str]]:
    """
    Convenience function to process all supported files in a directory without instantiating DatasetHandler.
    """
    handler = DatasetHandler(data_dir=data_dir, enable_cache=enable_cache)
    return handler.process_directory(dir_path, split_ratios)

def load_dataset_from_jsonl(
    dataset_name: str,
    split_name: str = "",
    data_dir: str = "./datasets",
    enable_cache: bool = True
) -> List[Dict]:
    """
    Convenience function to load a dataset from JSONL without instantiating DatasetHandler.
    """
    handler = DatasetHandler(data_dir=data_dir, enable_cache=enable_cache)
    return handler.load_dataset_from_jsonl(dataset_name, split_name)

def load_text_data(
    dataset_name: str,
    split_name: str = "",
    data_dir: str = "./datasets",
    enable_cache: bool = True
) -> List[str]:
    """
    Convenience function to load text data without instantiating DatasetHandler.
    """
    handler = DatasetHandler(data_dir=data_dir, enable_cache=enable_cache)
    return handler.load_text_data(dataset_name, split_name)

def analyze_text_lines(
    lines: List[str],
    top_n: int = 10,
    data_dir: str = "./datasets",
    enable_cache: bool = True
) -> Dict[str, Union[int, float, List[str]]]:
    """
    Convenience function to analyze text lines without instantiating DatasetHandler.
    """
    handler = DatasetHandler(data_dir=data_dir, enable_cache=enable_cache)
    return handler.analyze_text_lines(lines, top_n)

def sample_and_display_entries(
    dataset_name: str,
    split_name: str = "",
    num_samples: int = 5,
    data_dir: str = "./datasets",
    enable_cache: bool = True
) -> None:
    """
    Convenience function to sample and display entries without instantiating DatasetHandler.
    """
    handler = DatasetHandler(data_dir=data_dir, enable_cache=enable_cache)
    handler.sample_and_display_entries(dataset_name, split_name, num_samples)

def perform_exploratory_analysis(
    dataset_name: str,
    split_name: str = "",
    num_samples: int = 5,
    top_n: int = 10,
    data_dir: str = "./datasets",
    enable_cache: bool = True
) -> None:
    """
    Convenience function to perform an exploratory analysis without instantiating DatasetHandler.
    """
    handler = DatasetHandler(data_dir=data_dir, enable_cache=enable_cache)
    handler.perform_exploratory_analysis(dataset_name, split_name, num_samples, top_n)

if __name__ == "__main__":
    handler = DatasetHandler(enable_cache=True)

    print(Fore.CYAN + "\n=== Starting Dataset Processing and Validation ===" + Style.RESET_ALL)

    # 1. Process directory interactively
    while True:
        process_dir = input("\nWould you like to process a local directory? (y/n): ").lower()
        if process_dir == 'y':
            dir_path = input("Enter directory path: ")
            if os.path.isdir(dir_path):
                try:
                    splits = handler.process_directory(dir_path)
                    print(Fore.GREEN + f"\nSuccessfully processed directory: {dir_path}" + Style.RESET_ALL)
                    
                    # Validate processed splits
                    for split_name, texts in splits.items():
                        print(f"\nValidating {split_name} split ({len(texts)} texts):")
                        handler.sample_and_display_entries(dir_path, split_name, num_samples=3)
                        stats = handler.analyze_text_lines(texts, top_n=5)
                        print(f"Average length: {stats['avg_length']:.2f} chars")
                except Exception as e:
                    print(Fore.RED + f"Error processing directory: {str(e)}" + Style.RESET_ALL)
            else:
                print(Fore.RED + "Invalid directory path" + Style.RESET_ALL)
        elif process_dir == 'n':
            break
        else:
            print("Please enter 'y' or 'n'")

    # 2. Process standard datasets with comprehensive validation
    datasets = [
        ("openai_humaneval", None),
        ("mbpp", None), 
        ("text8", None),
        ("helpsteer2", {'train': 0.8, 'validation': 0.1, 'test': 0.1})
    ]

    for dataset_name, split_ratios in datasets:
        try:
            print(f"\nProcessing dataset: {dataset_name}")
            
            # Download and save dataset
            handler.download_and_save_dataset(dataset_name, split_ratios)

            # Test load_dataset_from_jsonl for JSON-based datasets
            if dataset_name in ["openai_humaneval", "mbpp", "helpsteer2"]:
                splits = ['train', 'validation', 'test'] if split_ratios else ['']
                for split in splits:
                    try:
                        data = handler.load_dataset_from_jsonl(dataset_name, split)
                        print(f"✓ Successfully loaded {len(data)} samples from {dataset_name} ({split if split else 'full'})")
                    except Exception as e:
                        print(Fore.RED + f"Error loading JSONL for {dataset_name} ({split}): {str(e)}" + Style.RESET_ALL)

            # Test load_text_data functionality
            splits = ['train', 'validation', 'test'] if split_ratios else ['']
            for split in splits:
                try:
                    lines = handler.load_text_data(dataset_name, split)
                    print(f"✓ Successfully loaded {len(lines)} text lines from {dataset_name} ({split if split else 'full'})")
                    
                    # Validate text content
                    if lines:
                        print("\nValidating text content:")
                        handler.sample_and_display_entries(dataset_name, split, num_samples=3)
                        stats = handler.analyze_text_lines(lines, top_n=5)
                        print(f"Text statistics validated - Avg length: {stats['avg_length']:.2f}")
                except Exception as e:
                    print(Fore.RED + f"Error loading text data for {dataset_name} ({split}): {str(e)}" + Style.RESET_ALL)

            # Comprehensive exploratory analysis
            for split in splits:
                print(f"\nPerforming full exploratory analysis on {dataset_name} - {split if split else 'full'} split:")
                try:
                    handler.perform_exploratory_analysis(
                        dataset_name,
                        split_name=split,
                        num_samples=5,
                        top_n=10
                    )
                except Exception as e:
                    print(Fore.RED + f"Error in exploratory analysis: {str(e)}" + Style.RESET_ALL)

            print(Fore.GREEN + f"✓ Successfully processed and validated {dataset_name}" + Style.RESET_ALL)

        except Exception as e:
            print(Fore.RED + f"Error processing {dataset_name}: {str(e)}" + Style.RESET_ALL)
            logger.error(f"Dataset {dataset_name} processing failed: {str(e)}")
            continue

    print(Fore.CYAN + "\n=== Dataset Processing and Validation Complete ===" + Style.RESET_ALL)
