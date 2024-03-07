import xml.etree.ElementTree as ET
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer


if not nltk.data.find('tokenizers/punkt'):
    nltk.download('punkt')


def load_toponyms(file_path: str) -> pd.DataFrame:
    """
    Loads a dataset from an XML file, extracting articles and their toponym information.

    Args:
        file_path (str): The path to the XML file.

    Returns:
        pd.DataFrame: A DataFrame containing the articles and extracted toponyms.
    """

    tree = ET.parse(file_path)
    root = tree.getroot()
    data = []

    for article in root.findall('article'):
        text = article.find('text').text

        for toponym in article.findall('.//toponym'):
            geonameid_str = None
            for tag in ['gaztag', 'geonamesID']:
                geoname_tag = toponym.find(tag)
                if geoname_tag is not None:
                    geonameid_str = geoname_tag.get('geonameid', geoname_tag.text)
                    break

            if geonameid_str is None or not geonameid_str.isdigit():
                continue

            geonameid = int(geonameid_str)
            mention = next((toponym.find(tag).text for tag in ['phrase', 'extractedName'] if toponym.find(tag) is not None), None)
            start = toponym.find('start').text
            end = toponym.find('end').text

            data.append([text, mention, start, end, geonameid])

    toponym_df = pd.DataFrame(data, columns=['text', 'mention', 'start', 'end', 'geonameid'])

    return toponym_df


def filter_toponyms(toponym_df: pd.DataFrame, gazetteer_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters a toponym DataFrame to include only rows where the corresponding 'geonameid' is present in the gazetteer DataFrame.

    Args:
        toponym_df (pd.DataFrame): The DataFrame containing toponyms with 'geonameid'.
        gazetteer_df (pd.DataFrame): The gazetteer DataFrame with 'geonameid' to filter against.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only matching 'geonameid' rows.
    """

    valid_geonameids = set(gazetteer_df['geonameid'])

    return toponym_df[toponym_df['geonameid'].isin(valid_geonameids)]


def truncate_text(text: str, start: int, end: int, tokenizer: AutoTokenizer, seq_length_limit: int) -> str:
    """
    Truncates the text so that it contains the toponym and stays within the sequence length limit.

    Args:
        text (str): The text to truncate.
        start (int): The start index of the toponym in the text.
        end (int): The end index of the toponym in the text.
        tokenizer (AutoTokenizer): The tokenizer to use for tokenizing the text.
        seq_length_limit (int): The maximum sequence length.

    Returns:
        str: The truncated text.
    """

    sentences = sent_tokenize(text)

    # Identify the index of the sentence containing the toponym
    toponym_sentence_index = next((i for i, sent in enumerate(sentences) if start < len(' '.join(sentences[:i+1]))), None)

    if toponym_sentence_index is None:
        return text

    # Tokenize sentences to count tokens for truncation logic
    tokenized_sentences = [tokenizer.tokenize(sent) for sent in sentences]
    tokens_before = sum(len(tokens) for tokens in tokenized_sentences[:toponym_sentence_index])
    tokens_containing = len(tokenized_sentences[toponym_sentence_index])
    tokens_after = sum(len(tokens) for tokens in tokenized_sentences[toponym_sentence_index+1:])

    before_index = 0
    after_index = len(sentences)

    # Adjust the number of sentences included based on token counts to meet the sequence length limit
    while tokens_before + tokens_containing + tokens_after > seq_length_limit and (before_index < toponym_sentence_index or after_index > toponym_sentence_index + 1):
        if tokens_before > tokens_after:
            tokens_before -= len(tokenized_sentences[before_index])
            before_index += 1
        else:
            tokens_after -= len(tokenized_sentences[after_index - 1])
            after_index -= 1

    return ' '.join(sentences[before_index:after_index])


def truncate_texts(toponym_df: pd.DataFrame, model_name: str, seq_length_limit: int) -> pd.DataFrame:
    """
    Truncates the texts in the DataFrame to the specified sequence length limit.

    Args:
        toponym_df (pd.DataFrame): The DataFrame containing the texts.
        model_name (str): The model name to load the tokenizer.
        seq_length_limit (int): The maximum sequence length.

    Returns:
        pd.DataFrame: The DataFrame with truncated texts added as a new column 'truncated_text'.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    truncate_func = lambda row: truncate_text(row['text'], int(row['start']), int(row['end']), tokenizer, seq_length_limit)
    toponym_df['truncated_text'] = toponym_df.apply(truncate_func, axis=1)

    return toponym_df