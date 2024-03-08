import os
import re
import unicodedata
import pandas as pd
import pickle
import requests
import inflect
from tqdm import tqdm


def normalize_toponym(toponym: str) -> str:
    """
    Normalizes a toponym by converting to lowercase, removing punctuation, and normalizing whitespace.

    Args:
        toponym (str): The toponym to normalize.

    Returns:
        str: The normalized toponym, with lowercase letters, no punctuation, and single spaces between words.
    """

    # Normalize Unicode characters to their closest ASCII representation
    toponym = unicodedata.normalize('NFKD', toponym)
    # Convert to lowercase
    toponym = toponym.lower()
    # Remove punctuation
    toponym = re.sub(r'[^\w\s]', '', toponym)
    # Normalize whitespace
    toponym = re.sub(r'\s+', ' ', toponym).strip()

    return toponym


def build_toponym_index(gazetteer_df: pd.DataFrame, file_path: str = 'toponym_index.pkl') -> dict:
    """
    Builds or loads a toponym index from a gazetteer DataFrame. If an index file exists, it loads from the file;
    otherwise, it builds the index and saves it.

    Args:
        gazetteer_df (pd.DataFrame): The gazetteer DataFrame.
        file_path (str): The path to save or load the toponym index.

    Returns:
        dict: The toponym index.
    """

    # Check if the index file already exists and load it if so
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    # Build the index from scratch if no file exists
    toponym_index = {}
    for row in tqdm(gazetteer_df.itertuples(index=False), total=len(gazetteer_df)):
        toponyms = [normalize_toponym(row.name)] + [normalize_toponym(toponym) for toponym in row.alternatenames.split(',')] if pd.notna(row.alternatenames) else [normalize_toponym(row.name)]
        for toponym in toponyms:
            if toponym in toponym_index:
                toponym_index[toponym].add(row.geonameid)
            else:
                toponym_index[toponym] = {row.geonameid}

    # Save the newly built index to the specified file
    with open(file_path, 'wb') as f:
        pickle.dump(toponym_index, f)

    return toponym_index


def extend_index_with_demonyms(toponym_index: dict, demonym_file_path: str) -> dict:
    """
    Extends the toponym index with demonyms by mapping each demonym to the same set of geonameids as its corresponding place.

    Args:
        toponym_index (dict): The existing toponym index.
        demonym_file_path (str): Path to the CSV file containing demonyms.

    Returns:
        dict: The updated toponym index.
    """

    demonym_df = pd.read_csv(demonym_file_path, header=None, names=['demonym', 'place'], keep_default_na=False)
    p = inflect.engine()

    for _, row in demonym_df.iterrows():
        singular = normalize_toponym(row['demonym'])
        plural = normalize_toponym(p.plural(singular))
        place = normalize_toponym(row['place'])

        if place in toponym_index:
            geonameids = toponym_index[place]
            toponym_index[singular] = geonameids
            toponym_index[plural] = geonameids

    return toponym_index


def geonames_api_call(toponym: str, username: str, mode: str, maxRows: int, fuzzyness: float = 0.0) -> set:
    """
    Makes an API call to Geonames to search for a toponym.

    Args:
        toponym (str): The toponym to search for.
        username (str): Geonames username.
        mode (str): Search mode, either 'regular' or 'fuzzy'.
        maxRows (int): Maximum number of rows to return.
        fuzzyness (float): Fuzziness parameter for the search.

    Returns:
        set: A set of geoname IDs.
    """
    
    url = "http://api.geonames.org/searchJSON"
    params = {
        "q": toponym,
        "username": username,
        "maxRows": maxRows
    }

    if mode == 'fuzzy':
        params["fuzzy"] = fuzzyness

    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise GeonamesAPIError(f"Geonames API call failed with status code {response.status_code}: {response.text}")

    geoname_ids = {item['geonameId'] for item in response.json().get('geonames', [])}

    return geoname_ids


def generate_candidates(toponym_df: pd.DataFrame, toponym_index: dict, username: str = 'demo', cache_file: str = 'geonames_cache.pkl', maxRows: int = 50, fuzzyness: float = 0.4) -> pd.DataFrame:
    """
    Generates a set of candidate locations for each toponym in the toponym dataset.
    Performs both regular and fuzzy searches if the index yields no results.

    Args:
        toponym_df (pd.DataFrame): The toponym dataset DataFrame.
        toponym_index (dict): The toponym index.
        username (str): Geonames username.
        cache_file (str): Path to the cache file.
        maxRows (int): Maximum number of rows for Geonames API calls.
        fuzzyness (float): Fuzziness parameter for fuzzy searches.

    Returns:
        pd.DataFrame: The toponym dataset DataFrame with a new "candidates" column.
    """

    # Load or initialize the cache
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
    else:
        cache = {}

    toponym_df['candidates'] = None

    for index, row in toponym_df.iterrows():
        normalized_mention = normalize_toponym(row['mention'])
        candidates = toponym_index.get(normalized_mention, None)

        if candidates is None:
            # Attempt to retrieve candidates from cache or perform API calls if necessary
            regular_cache_key = (normalized_mention, 'regular', maxRows)
            fuzzy_cache_key = (normalized_mention, 'fuzzy', maxRows, fuzzyness)

            regular_candidates = cache.get(regular_cache_key, None)
            if regular_candidates is None:
                regular_candidates = geonames_api_call(normalized_mention, username=username, mode='regular', maxRows=maxRows)
                cache[regular_cache_key] = regular_candidates

            fuzzy_candidates = cache.get(fuzzy_cache_key, None)
            if fuzzy_candidates is None:
                fuzzy_candidates = geonames_api_call(normalized_mention, username=username, mode='fuzzy', maxRows=maxRows, fuzzyness=fuzzyness)
                cache[fuzzy_cache_key] = fuzzy_candidates

            # Combine and filter out duplicates
            candidates = regular_candidates.union(fuzzy_candidates)

        toponym_df.at[index, 'candidates'] = list(candidates)

    # Save the updated cache
    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)

    return toponym_df