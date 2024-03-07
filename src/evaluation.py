import pandas as pd
import numpy as np
import torch
from typing import List
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from sklearn.metrics.pairwise import cosine_similarity
from haversine import haversine
import random
import os


def predict_locations(toponym_df: pd.DataFrame, gazetteer_df: pd.DataFrame, model: SentenceTransformer, gazetteer_embeddings: torch.Tensor = None, batch_size: int = 16) -> pd.DataFrame:
    """
    Generates embeddings for toponyms and gazetteer entries, and disambiguates each toponym by identifying
    the candidate with the highest cosine similarity to the toponym embedding. Reuses gazetteer embeddings if provided.

    Args:
        toponym_df (pd.DataFrame): DataFrame containing toponyms and their candidate geonameids.
        gazetteer_df (pd.DataFrame): DataFrame containing gazetteer data.
        model (SentenceTransformer): The SentenceTransformer model to use for generating embeddings.
        gazetteer_embeddings (torch.Tensor, optional): Precomputed embeddings for the gazetteer data.

    Returns:
        pd.DataFrame: DataFrame with the disambiguation predictions added.
    """

    # Work on a copy of the DataFrame to avoid SettingWithCopyWarning
    toponym_df = toponym_df.copy()

    # Initialize the column for predicted geonameid
    toponym_df['predicted_geonameid'] = None

    # Generate toponym embeddings
    print("Generating toponym embeddings...")
    toponym_embeddings = model.encode(toponym_df['truncated_text'].tolist(), convert_to_tensor=True, show_progress_bar=True)

    # Generate or reuse gazetteer embeddings
    if gazetteer_embeddings is None:
        print("Generating gazetteer embeddings...")
        gazetteer_embeddings = model.encode(gazetteer_df['pseudotext'].tolist(), batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)

    # Map each geonameid to its corresponding embedding
    gazetteer_dict = dict(zip(gazetteer_df['geonameid'], gazetteer_embeddings))

    for toponym_embedding, (_, row) in zip(toponym_embeddings, toponym_df.iterrows()):
        toponym_embedding = toponym_embedding.unsqueeze(0).cpu()
        candidate_embeddings = torch.stack([gazetteer_dict[cand] for cand in row['candidates'] if cand in gazetteer_dict]).cpu()

        if len(candidate_embeddings) == 0:
            toponym_df.at[row.name, 'predicted_geonameid'] = None
            continue

        cos_similarities = cosine_similarity(toponym_embedding, candidate_embeddings)[0]
        most_likely_candidate = row['candidates'][np.argmax(cos_similarities)]
        toponym_df.at[row.name, 'predicted_geonameid'] = most_likely_candidate

    return toponym_df


def calculate_distances(toponym_df: pd.DataFrame, gazetteer_df: pd.DataFrame) -> np.ndarray:
    """
    Calculate the distances between the actual and predicted locations of toponyms.

    Args:
        toponym_df (pd.DataFrame): DataFrame containing toponyms with actual and predicted geoname IDs.
        gazetteer_df (pd.DataFrame): DataFrame containing the coordinates for each geoname ID.

    Returns:
        np.ndarray: An array of distances in kilometers.
    """

    # Maximum error represented as half the Earth's equator length in kilometers
    MAX_ERROR = 20039

    # Create a dictionary mapping geoname IDs to their coordinates for quick lookup
    gazetteer_dict = gazetteer_df.set_index('geonameid')[['latitude', 'longitude']].to_dict('index')

    distances = []
    for _, row in toponym_df.iterrows():
        actual_coords = gazetteer_dict.get(row['geonameid'], None)
        predicted_coords = gazetteer_dict.get(row['predicted_geonameid'], None)

        if actual_coords and predicted_coords:
            distance = haversine((actual_coords['latitude'], actual_coords['longitude']), 
                                 (predicted_coords['latitude'], predicted_coords['longitude']))
            distances.append(distance)
        else:
            # Assign max error value if coordinates are missing
            distances.append(MAX_ERROR)

    return np.array(distances)


def evaluate_predictions(toponym_df: pd.DataFrame, gazetteer_df: pd.DataFrame) -> dict:
    """
    Evaluate the toponym predictions using various metrics, including accuracy, accuracy@161km,
    mean error distance, area under the curve, correct country, and correct admin1.
    Correct country and admin1 are assumed to be accurate if the predicted geonameid matches the actual geonameid.

    Args:
        toponym_df (pd.DataFrame): DataFrame containing toponyms with actual and predicted geoname IDs.
        gazetteer_df (pd.DataFrame): DataFrame containing the coordinates and other info for each geoname ID.

    Returns:
        dict: A dictionary containing the evaluation metrics.
    """

    distances = calculate_distances(toponym_df, gazetteer_df)

    # Accuracy based on matching geonameid
    accuracy = np.mean(toponym_df['geonameid'] == toponym_df['predicted_geonameid'])

    # Accuracy at 161 km
    accuracy_at_161 = np.mean(distances <= 161)

    # Mean error distance
    mean_error_distance = np.mean(distances)

    # Area Under the Curve (AUC)
    adjusted_distances = distances + 1  # To avoid log(0)
    ln_distances = np.log(adjusted_distances)
    MAX_ERROR = 20039
    auc = np.trapz(sorted(ln_distances)) / (np.log(MAX_ERROR) * (len(ln_distances) - 1))

    # Correct Country and Correct Admin1 accuracy
    gazetteer_dict = gazetteer_df.set_index('geonameid')[['country_name', 'admin1_name']].to_dict('index')
    country_matches = admin1_matches = 0

    for _, row in toponym_df.iterrows():
        if row['geonameid'] == row['predicted_geonameid']:
            country_matches += 1
            admin1_matches += 1
        else:
            actual_info = gazetteer_dict.get(row['geonameid'])
            predicted_info = gazetteer_dict.get(row['predicted_geonameid'])

            if actual_info and predicted_info:
                if actual_info['country_name'] == predicted_info['country_name']:
                    country_matches += 1
                if actual_info['admin1_name'] == predicted_info['admin1_name']:
                    admin1_matches += 1

    correct_country_accuracy = country_matches / len(toponym_df)
    correct_admin1_accuracy = admin1_matches / len(toponym_df)

    return {
        'Accuracy': accuracy,
        'Accuracy@161km': accuracy_at_161,
        'Mean Error Distance': mean_error_distance,
        'Area Under the Curve': auc,
        'Correct Country': correct_country_accuracy,
        'Correct Admin1': correct_admin1_accuracy
    }


def evaluate_model(test_dfs: List[pd.DataFrame], test_df_names: List[str], gazetteer_df: pd.DataFrame, model: SentenceTransformer, model_name: str, batch_size: int = 16) -> None:
    """
    Evaluates the model on multiple test datasets, prints the evaluation results for each dataset.

    Args:
        test_dfs (list of pd.DataFrame): List of test DataFrames.
        test_df_names (list of str): List of names corresponding to the test DataFrames.
        gazetteer_df (pd.DataFrame): DataFrame containing gazetteer data.
        model (SentenceTransformer): The SentenceTransformer model used for predictions.
        model_name (str): The name of the SentenceTransformer model used for predictions.
    """

    # Generate gazetteer embeddings once for all datasets
    print("Generating gazetteer embeddings...")
    gazetteer_embeddings = model.encode(gazetteer_df['pseudotext'].tolist(), batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)

    for test_df, name in zip(test_dfs, test_df_names):
        # Predict locations
        predictions_df = predict_locations(test_df, gazetteer_df, model, gazetteer_embeddings, batch_size=batch_size)

        # Evaluate predictions
        evaluation_metrics = evaluate_predictions(predictions_df, gazetteer_df)

        # Print evaluation results
        print(f"Test Results for model {model_name} ({name}):")
        for metric, value in evaluation_metrics.items():
            formatted_value = f"{value:.3f}" if metric != "Mean Error Distance" else f"{int(value)} km"
            print(f"  {metric}: {formatted_value}")
        print("\n")


def evaluate_population(test_dfs: List[pd.DataFrame], test_df_names: List[str], gazetteer_df: pd.DataFrame) -> None:
    """
    Evaluates the population heuristic on multiple test datasets.

    Args:
        test_dfs (list of pd.DataFrame): List of test DataFrames.
        test_df_names (list of str): Names corresponding to the test DataFrames.
        gazetteer_df (pd.DataFrame): DataFrame containing gazetteer data.
    """

    gazetteer_dict = gazetteer_df.set_index('geonameid')['population'].to_dict()

    for test_df, name in zip(test_dfs, test_df_names):
        # Create a copy to avoid modifying the original DataFrame
        predictions_df = test_df.copy()
        
        # Apply population heuristic
        predictions_df['predicted_geonameid'] = predictions_df['candidates'].apply(
            lambda candidates: max(candidates, key=lambda x: gazetteer_dict.get(x, 0))
        )

        # Evaluate predictions
        evaluation_metrics = evaluate_predictions(predictions_df, gazetteer_df)

        # Print evaluation results
        print(f"Population Heuristic Results ({name}):")
        for metric, value in evaluation_metrics.items():
            formatted_value = f"{value:.3f}" if metric != "Mean Error Distance" else f"{int(value)} km"
            print(f"  {metric}: {formatted_value}")
        print("\n")


def evaluate_random(test_dfs: List[pd.DataFrame], test_df_names: List[str], gazetteer_df: pd.DataFrame) -> None:
    """
    Evaluates the random selection heuristic on multiple test datasets.

    Args:
        test_dfs (list of pd.DataFrame): List of test DataFrames.
        test_df_names (list of str): Names corresponding to the test DataFrames.
        gazetteer_df (pd.DataFrame): DataFrame containing gazetteer data.
    """

    for test_df, name in zip(test_dfs, test_df_names):
        # Create a copy to avoid modifying the original DataFrame
        predictions_df = test_df.copy()
        
        # Apply random selection heuristic
        predictions_df['predicted_geonameid'] = predictions_df['candidates'].apply(
            lambda candidates: random.choice(candidates) if candidates else None
        )

        # Evaluate predictions
        evaluation_metrics = evaluate_predictions(predictions_df, gazetteer_df)

        # Print evaluation results
        print(f"Random Selection Heuristic Results ({name}):")
        for metric, value in evaluation_metrics.items():
            formatted_value = f"{value:.3f}" if metric != "Mean Error Distance" else f"{int(value)} km"
            print(f"  {metric}: {formatted_value}")
        print("\n")


def print_incorrect_predictions(test_dfs: List[pd.DataFrame], test_df_names: List[str], gazetteer_df: pd.DataFrame, model: SentenceTransformer, batch_size: int = 16) -> None:
    """
    Prints details of incorrect predictions for each test dataset, including whether the actual
    geonameid is among the candidates.

    Args:
        test_dfs (list of pd.DataFrame): List of test DataFrames.
        test_df_names (list of str): List of names corresponding to the test DataFrames.
        gazetteer_df (pd.DataFrame): DataFrame containing gazetteer data.
        model (SentenceTransformer): The SentenceTransformer model used for predictions.
        batch_size (int): The batch size for encoding.
    """

    gazetteer_dict = gazetteer_df.set_index('geonameid')['pseudotext'].to_dict()

    print("Generating gazetteer embeddings...")
    gazetteer_embeddings = model.encode(gazetteer_df['pseudotext'].tolist(), batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)

    for test_df, name in zip(test_dfs, test_df_names):
        predictions_df = predict_locations(test_df, gazetteer_df, model, gazetteer_embeddings, batch_size=batch_size)
        wrong_predictions = predictions_df[predictions_df['geonameid'] != predictions_df['predicted_geonameid']]

        print(f"\nIncorrect Predictions for {name}:")
        for _, row in wrong_predictions.iterrows():
            actual_pseudotext = gazetteer_dict.get(row['geonameid'], "N/A")
            predicted_pseudotext = gazetteer_dict.get(row['predicted_geonameid'], "N/A")
            actual_in_candidates = row['geonameid'] in row['candidates']
            candidate_status = "Actual ID in candidates" if actual_in_candidates else "Actual ID not in candidates"

            print(f"\nToponym: {row['mention']}")
            print(candidate_status)
            print(f"Actual GeonameID: {row['geonameid']} - Pseudotext: {actual_pseudotext}")
            print(f"Predicted GeonameID: {row['predicted_geonameid']} - Pseudotext: {predicted_pseudotext}")


class ToponymResolutionEvaluator(SentenceEvaluator):
    """
    Custom evaluator for toponym resolution that evaluates a SentenceTransformer model
    based on accuracy, accuracy@161km, mean error distance, and AUC. The results are 
    continuously written to a CSV file named after the model and stored in a folder 
    named 'evaluation_results'.

    Args:
        gazetteer_df (pd.DataFrame): DataFrame containing gazetteer data.
        eval_df (pd.DataFrame): DataFrame containing evaluation data.
        model_name (str): Name of the SentenceTransformer model used for evaluation.
        batch_size (int): Batch size used for encoding.
    """
    
    def __init__(self, gazetteer_df, eval_df, model_name, batch_size=16):
        self.gazetteer_df = gazetteer_df
        self.eval_df = eval_df
        self.model_name = model_name
        self.batch_size = batch_size
        self.results_folder = "evaluation_results"
        os.makedirs(self.results_folder, exist_ok=True)

    def __call__(self, model: SentenceTransformer, output_path: str = None, epoch: int = -1, steps: int = -1):
        """
        Generates predictions and evaluates the SentenceTransformer model on the evaluation set.

        Args:
            model (SentenceTransformer): The model to be evaluated.
            output_path (str, optional): The path where evaluation results are stored.
            epoch (int, optional): The current epoch number.
            steps (int, optional): The current step number in the current epoch.

        Returns:
            float: The accuracy of the model on the evaluation set.
        """

        print("Generating gazetteer embeddings...")
        gazetteer_embeddings = model.encode(self.gazetteer_df['pseudotext'].tolist(), batch_size=self.batch_size, convert_to_tensor=True, show_progress_bar=True)

        # Predict locations for the evaluation set
        predictions_df = predict_locations(self.eval_df, self.gazetteer_df, model, gazetteer_embeddings, batch_size=self.batch_size)

        # Evaluate predictions
        evaluation_metrics = evaluate_predictions(predictions_df, self.gazetteer_df)

        # Print evaluation results
        print(f"Evaluation Results for model {self.model_name} (Epoch {int(epoch) + 1}, Step {int(steps)}):")
        for metric, value in evaluation_metrics.items():
            formatted_value = f"{value:.3f}" if metric != "Mean Error Distance" else f"{int(value)} km"
            print(f"  {metric}: {formatted_value}")
        print("\n")

        evaluation_metrics['epoch'] = int(epoch) + 1
        evaluation_metrics['steps'] = int(steps)

        # Append evaluation results to a CSV file
        results_filename = f"{self.model_name}.csv"
        results_filepath = os.path.join(self.results_folder, results_filename)
        with open(results_filepath, 'a') as file:
            pd.DataFrame([evaluation_metrics]).to_csv(file, header=file.tell()==0, index=False)

        return evaluation_metrics['Accuracy']