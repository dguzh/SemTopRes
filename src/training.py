import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
from torch.utils.data import DataLoader
from sentence_transformers import InputExample


def split_train_eval_test(lgl_df: pd.DataFrame, gwn_df: pd.DataFrame, trn_df: pd.DataFrame, test_size: float = 0.2, eval_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Splits the combined datasets into train, eval, and test sets, ensuring that the same text 
    does not appear in multiple sets.

    Args:
        lgl_df (pd.DataFrame): DataFrame for LGL dataset.
        gwn_df (pd.DataFrame): DataFrame for GWN dataset.
        trn_df (pd.DataFrame): DataFrame for TRN dataset.
        test_size (float): Proportion of dataset to include in test split.
        eval_size (float): Proportion of dataset to include in eval split.

    Returns:
        tuple: Tuple containing train, eval, and test DataFrames.
    """

    combined_df = pd.concat([lgl_df, gwn_df, trn_df])
    grouped = combined_df.groupby('text')
    unique_texts = list(grouped.groups.keys())

    train_texts, test_texts = train_test_split(unique_texts, test_size=test_size + eval_size, random_state=42)

    if eval_size > 0:
        eval_size_adjusted = eval_size / (test_size + eval_size)  # Adjust eval size proportion
        eval_texts, test_texts = train_test_split(test_texts, test_size=1-eval_size_adjusted, random_state=42)
    else:
        eval_texts = []

    filter_df = lambda df, texts: df[df['text'].isin(texts)]
    train_df = filter_df(combined_df, train_texts)
    eval_df = filter_df(combined_df, eval_texts)

    test_lgl_df = filter_df(lgl_df, test_texts)
    test_gwn_df = filter_df(gwn_df, test_texts)
    test_trn_df = filter_df(trn_df, test_texts)

    return train_df, eval_df, (test_lgl_df, test_gwn_df, test_trn_df)


def create_dataloader(train_df: pd.DataFrame, gazetteer_df: pd.DataFrame, batch_size: int = 16) -> DataLoader:
    """
    Creates a DataLoader for training, containing positive and negative pairs of text and pseudotext.

    Args:
        train_df (pd.DataFrame): The DataFrame containing the training data.
        gazetteer_df (pd.DataFrame): The DataFrame containing the gazetteer data.
        batch_size (int): The batch size for the DataLoader.

    Returns:
        DataLoader: A DataLoader containing InputExamples for training.
    """

    # Create a dictionary for faster lookup of pseudotexts
    pseudotext_dict = gazetteer_df.set_index('geonameid')['pseudotext'].to_dict()

    train_examples = []
    for index, row in train_df.iterrows():
        correct_pseudotext = pseudotext_dict.get(row['geonameid'])

        # Positive pair
        positive_example = InputExample(texts=[row['truncated_text'], correct_pseudotext], label=1.0)
        train_examples.append(positive_example)

        # Negative pairs
        incorrect_candidates = [cand for cand in row['candidates'] if cand != row['geonameid']]
        negative_examples = [InputExample(texts=[row['truncated_text'], pseudotext_dict[cand]], label=0.0) for cand in incorrect_candidates]
        train_examples.extend(negative_examples)

    return DataLoader(train_examples, shuffle=True, batch_size=batch_size)