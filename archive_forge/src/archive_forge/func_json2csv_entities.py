import csv
import gzip
import json
from nltk.internals import deprecated
def json2csv_entities(tweets_file, outfile, main_fields, entity_type, entity_fields, encoding='utf8', errors='replace', gzip_compress=False):
    """
    Extract selected fields from a file of line-separated JSON tweets and
    write to a file in CSV format.

    This utility function allows a file of full Tweets to be easily converted
    to a CSV file for easier processing of Twitter entities. For example, the
    hashtags or media elements of a tweet can be extracted.

    It returns one line per entity of a Tweet, e.g. if a tweet has two hashtags
    there will be two lines in the output file, one per hashtag

    :param tweets_file: the file-like object containing full Tweets

    :param str outfile: The path of the text file where results should be        written

    :param list main_fields: The list of fields to be extracted from the main        object, usually the tweet. Useful examples: 'id_str' for the tweetID. See        <https://dev.twitter.com/overview/api/tweets> for a full list of fields.
        e. g.: ['id_str'], ['id', 'text', 'favorite_count', 'retweet_count']
        If `entity_type` is expressed with hierarchy, then it is the list of        fields of the object that corresponds to the key of the entity_type,        (e.g., for entity_type='user.urls', the fields in the main_fields list        belong to the user object; for entity_type='place.bounding_box', the        files in the main_field list belong to the place object of the tweet).

    :param list entity_type: The name of the entity: 'hashtags', 'media',        'urls' and 'user_mentions' for the tweet object. For a user object,        this needs to be expressed with a hierarchy: `'user.urls'`. For the        bounding box of the Tweet location, use `'place.bounding_box'`.

    :param list entity_fields: The list of fields to be extracted from the        entity. E.g. `['text']` (of the Tweet)

    :param error: Behaviour for encoding errors, see        https://docs.python.org/3/library/codecs.html#codec-base-classes

    :param gzip_compress: if `True`, output files are compressed with gzip
    """
    writer, outf = _outf_writer(outfile, encoding, errors, gzip_compress)
    header = get_header_field_list(main_fields, entity_type, entity_fields)
    writer.writerow(header)
    for line in tweets_file:
        tweet = json.loads(line)
        if _is_composed_key(entity_type):
            key, value = _get_key_value_composed(entity_type)
            object_json = _get_entity_recursive(tweet, key)
            if not object_json:
                continue
            object_fields = extract_fields(object_json, main_fields)
            items = _get_entity_recursive(object_json, value)
            _write_to_file(object_fields, items, entity_fields, writer)
        else:
            tweet_fields = extract_fields(tweet, main_fields)
            items = _get_entity_recursive(tweet, entity_type)
            _write_to_file(tweet_fields, items, entity_fields, writer)
    outf.close()