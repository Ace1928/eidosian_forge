import importlib
import json
import time
import datetime
import os
import requests
import shutil
import hashlib
import tqdm
import math
import zipfile
import parlai.utils.logging as logging
def download_multiprocess(urls, path, num_processes=32, chunk_size=100, dest_filenames=None, error_path=None):
    """
    Download items in parallel (e.g. for an image + dialogue task).

    WARNING: may have issues with OS X.

    :param urls:
        Array of urls to download
    :param path:
        directory to save items in
    :param num_processes:
        number of processes to use
    :param chunk_size:
        chunk size to use
    :param dest_filenames:
        optional array of same length as url with filenames.  Images will be
        saved as path + dest_filename
    :param error_path:
        where to save error logs
    :return:
        array of tuples of (destination filename, http status code, error
        message if any). Note that upon failure, file may not actually be
        created.
    """
    pbar = tqdm.tqdm(total=len(urls), position=0)
    if dest_filenames:
        if len(dest_filenames) != len(urls):
            raise Exception('If specified, destination filenames must equal url array in length.')
    else:

        def _naming_fn(url, url_metadata=None):
            return hashlib.md5(url.encode('utf-8')).hexdigest()
        dest_filenames = [_naming_fn(url) for url in urls]
    items = zip(urls, dest_filenames)
    remaining_items = [it for it in items if not os.path.isfile(os.path.join(path, it[1]))]
    logging.info(f'Of {len(urls)} items, {len(urls) - len(remaining_items)} already existed; only going to download {len(remaining_items)} items.')
    pbar.update(len(urls) - len(remaining_items))
    pool_chunks = ((remaining_items[i:i + chunk_size], path, _download_multiprocess_single) for i in range(0, len(remaining_items), chunk_size))
    remaining_chunks_count = math.ceil(float(len(remaining_items) / chunk_size))
    logging.info(f'Going to download {remaining_chunks_count} chunks with {chunk_size} images per chunk using {num_processes} processes.')
    pbar.desc = 'Downloading'
    all_results = []
    collected_errors = []
    with Pool(num_processes) as pool:
        for idx, chunk_result in enumerate(pool.imap_unordered(_download_multiprocess_map_chunk, pool_chunks, 2)):
            all_results.extend(chunk_result)
            for dest_file, http_status_code, error_msg in chunk_result:
                if http_status_code != 200:
                    collected_errors.append({'dest_file': dest_file, 'status_code': http_status_code, 'error': error_msg})
                    logging.error(f'Bad download - chunk: {idx}, dest_file: {dest_file}, http status code: {http_status_code}, error_msg: {error_msg}')
            pbar.update(len(chunk_result))
    pbar.close()
    if error_path:
        now = time.strftime('%Y%m%d-%H%M%S')
        error_filename = os.path.join(error_path, 'parlai_download_multiprocess_errors_%s.log' % now)
        with open(os.path.join(error_filename), 'w+') as error_file:
            error_file.write(json.dumps(collected_errors))
            logging.error(f'Summary of errors written to {error_filename}')
    logging.info(f'Of {len(remaining_items)} items attempted downloading, {len(collected_errors)} had errors.')
    logging.debug('Finished downloading chunks.')
    return all_results