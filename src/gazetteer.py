import pandas as pd
import numpy as np


def load_gazetteer(file_path: str) -> pd.DataFrame:
    """
    Loads gazetteer data from a tab-delimited text file into a DataFrame, processing in chunks,
    excluding unnecessary columns, assuming the file does not have headers.

    Args:
        file_path (str): The path to the tab-delimited text file.

    Returns:
        pd.DataFrame: A DataFrame containing the gazetteer data, minus the excluded columns.
    """

    # Full list of columns based on the file's structure
    FULL_COLS = ['geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 'longitude', 
                 'feature_class', 'feature_code', 'country_code', 'cc2', 'admin1_code', 'admin2_code', 
                 'admin3_code', 'admin4_code', 'population', 'elevation', 'dem', 'timezone', 
                 'modification_date']
    
    # Columns to keep
    COLS_TO_LOAD = ['geonameid', 'name', 'alternatenames', 'latitude', 'longitude', 
                 'feature_class', 'feature_code', 'country_code', 'admin1_code',
                 'admin2_code', 'admin3_code', 'admin4_code', 'population']

    DTYPE = {
        'geonameid': 'int32',
        'population': 'int64',
        'latitude': 'float32',
        'longitude': 'float32'
    }

    chunk_size = 1000000  # Adjust based on your system's memory capacity.
    chunks = pd.read_csv(file_path, delimiter='\t', names=FULL_COLS, low_memory=False, 
                         usecols=COLS_TO_LOAD, dtype=DTYPE, keep_default_na=False, chunksize=chunk_size)

    # Concatenate chunks into one DataFrame
    gazetteer_df = pd.concat(chunks, ignore_index=True)

    # Convert GeoName codes into categories
    category_columns = ['feature_class', 'feature_code', 'country_code', 'admin1_code', 'admin2_code', 'admin3_code', 'admin4_code']
    for col in category_columns:
        gazetteer_df[col] = gazetteer_df[col].astype('category')

    return gazetteer_df


def filter_gazetteer(gazetteer_df: pd.DataFrame, toponym_dfs: list) -> pd.DataFrame:
    """
    Filters the gazetteer DataFrame to include only entries that are resolved locations 
    or candidates of toponyms in provided toponym dataframes.

    Args:
        gazetteer_df (pd.DataFrame): The gazetteer DataFrame.
        toponym_dfs (list): A list of toponym DataFrames.

    Returns:
        pd.DataFrame: The filtered gazetteer DataFrame.
    """

    # Extracting geonameid sets from each toponym DataFrame
    geonameid_sets = []
    for df in toponym_dfs:
        resolved_ids = df['geonameid'].dropna().unique()  # Labelled locations
        candidate_ids = df['candidates'].dropna().explode().unique()  # Candidate locations
        geonameid_sets.extend([resolved_ids, candidate_ids])

    # Combine all geonameid sets into one unique set for filtering
    all_geonameids = set().union(*geonameid_sets)

    # Filter gazetteer DataFrame
    gazetteer_df = gazetteer_df[gazetteer_df['geonameid'].isin(all_geonameids)]

    return gazetteer_df


def load_lookup_tables(admin1_path: str, admin2_path: str, country_info_path: str, feature_codes_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads lookup tables from specified file paths.

    Args:
        admin1_path (str): Path to the admin1 lookup table.
        admin2_path (str): Path to the admin2 lookup table.
        country_info_path (str): Path to the country info lookup table.
        feature_codes_path (str): Path to the feature codes lookup table.

    Returns:
        tuple: Tuple containing DataFrames for admin1, admin2, country info, and feature codes.
    """

    admin1_df = pd.read_csv(admin1_path, sep='\t', header=None, 
                            names=['admin1_code', 'admin1_name', 'ascii_name', 'admin1_geonameid'], 
                            na_values=None, dtype={'admin1_geonameid': 'Int64'})
    admin2_df = pd.read_csv(admin2_path, sep='\t', header=None, 
                            names=['admin2_code', 'admin2_name', 'ascii_name', 'admin2_geonameid'], 
                            na_values=None, dtype={'admin2_geonameid': 'Int64'})
    country_df = pd.read_csv(country_info_path, sep='\t', header=None, skiprows=50,  # Adjust skiprows as needed to skip actual comment lines
                             names=['country_code', 'ISO3', 'ISO-Numeric', 'fips', 'country_name', 
                                    'Capital', 'Area(in sq km)', 'Population', 'Continent', 'tld', 
                                    'CurrencyCode', 'CurrencyName', 'Phone', 'Postal Code Format', 
                                    'Postal Code Regex', 'Languages', 'country_geonameid', 'neighbours', 
                                    'EquivalentFipsCode'],
                             na_values=None, dtype={'country_geonameid': 'Int64'})
    feature_df = pd.read_csv(feature_codes_path, sep='\t', header=None, 
                             names=['feature_code', 'feature_name', 'feature_description'], 
                             na_values=None)

    return admin1_df, admin2_df, country_df, feature_df


def generate_descriptor_names(gazetteer_df: pd.DataFrame, admin1_path: str, admin2_path: str, country_info_path: str, feature_codes_path: str) -> pd.DataFrame:
    """
    Merges lookup tables with the gazetteer DataFrame.

    Args:
        gazetteer_df (pd.DataFrame): The gazetteer DataFrame.
        admin1_path (str): Path to the admin1 lookup table.
        admin2_path (str): Path to the admin2 lookup table.
        country_info_path (str): Path to the country info lookup table.
        feature_codes_path (str): Path to the feature codes lookup table.

    Returns:
        pd.DataFrame: The enriched gazetteer DataFrame with descriptive names for admin1, admin2, countries, and feature codes.
    """

    admin1_df, admin2_df, country_df, feature_df = load_lookup_tables(admin1_path, admin2_path, country_info_path, feature_codes_path)

    # Construct full codes for admin1, admin2, and feature codes for merging
    gazetteer_df = gazetteer_df.assign(
        full_admin1_code=lambda x: np.where((x['admin1_code'] != '00') & pd.notna(x['admin1_code']) & pd.notna(x['country_code']), x['country_code'] + '.' + x['admin1_code'], np.nan),
        full_admin2_code=lambda x: np.where((x['admin2_code'] != '00') & pd.notna(x['admin2_code']) & pd.notna(x['admin1_code']) & pd.notna(x['country_code']), x['country_code'] + '.' + x['admin1_code'] + '.' + x['admin2_code'], np.nan),
        full_feature_code=lambda x: np.where(pd.notna(x['feature_class']) & pd.notna(x['feature_code']), x['feature_class'] + '.' + x['feature_code'], np.nan)
    )

    # Merge with lookup tables and drop any unnecessary columns after merge
    gazetteer_df = gazetteer_df.merge(admin1_df[['admin1_code', 'admin1_name', 'admin1_geonameid']], left_on='full_admin1_code', right_on='admin1_code', how='left', suffixes=('', '_drop')).filter(regex='^(?!.*_drop$).*')
    gazetteer_df = gazetteer_df.merge(admin2_df[['admin2_code', 'admin2_name', 'admin2_geonameid']], left_on='full_admin2_code', right_on='admin2_code', how='left', suffixes=('', '_drop')).filter(regex='^(?!.*_drop$).*')
    gazetteer_df = gazetteer_df.merge(country_df[['country_code', 'country_name', 'country_geonameid']], left_on='country_code', right_on='country_code', how='left', suffixes=('', '_drop')).filter(regex='^(?!.*_drop$).*')
    gazetteer_df = gazetteer_df.merge(feature_df[['feature_code', 'feature_name']], left_on='full_feature_code', right_on='feature_code', how='left', suffixes=('', '_drop')).filter(regex='^(?!.*_drop$).*')

    # Correcting the country name assignment for NaN country codes (otherwise it resolves to "Namibia")
    gazetteer_df['country_name'] = np.where(gazetteer_df['country_code'].isna(), np.nan, gazetteer_df['country_name'])

    # Drop the temporary full code columns used for merging
    gazetteer_df.drop(columns=['full_admin1_code', 'full_admin2_code', 'full_feature_code'], inplace=True)

    return gazetteer_df


def generate_pseudotext(row: pd.Series) -> str:
    """
    Generates a pseudotext description for a single row in the gazetteer DataFrame.

    Args:
        row (pd.Series): A row from the gazetteer DataFrame.

    Returns:
        str: The pseudotext for the location, formatted as "Name (Feature) in Admin2, Admin1, Country".
    """

    # Initialize the description with the primary name
    components = [row['name']]

    # Append administrative and country names if they are not null
    for field in ['admin2_name', 'admin1_name', 'country_name']:
        if pd.notna(row[field]):
            components.append(row[field])

    location_str = " in " + ", ".join(components[1:]) if len(components) > 1 else ""
    feature_str = f" ({row['feature_name']})" if pd.notna(row['feature_name']) else ""
    
    return f"{components[0]}{feature_str}{location_str}"


def generate_pseudotexts(gazetteer_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates pseudotext descriptions for each entry in the gazetteer DataFrame.

    Args:
        gazetteer_df (pd.DataFrame): The gazetteer DataFrame.

    Returns:
        pd.DataFrame: The gazetteer DataFrame with an additional 'pseudotext' column.
    """

    gazetteer_df['pseudotext'] = gazetteer_df.apply(generate_pseudotext, axis=1)

    return gazetteer_df