from kedro.pipeline import *
from kedro.io import *
from kedro.runner import *
import pandas as pd
import numpy as np
import pickle
import os
import math


def replace_empty_strings_with_none(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Replace empty strings with None values
    Args:
        raw_data:

    Returns:
        raw_data with empty strings replaced with None values
    """
    # Replace empty strings with None values
    processed_raw_data = raw_data.replace(' ', np.nan)

    # Replace -98, -99 with None values
    processed_raw_data = processed_raw_data.replace(-99, np.nan)
    processed_raw_data = processed_raw_data.replace(-98, np.nan)
    processed_raw_data = processed_raw_data.replace(-99.0, np.nan)
    processed_raw_data = processed_raw_data.replace(-98.0, np.nan)
    processed_raw_data = processed_raw_data.replace(-99., np.nan)
    processed_raw_data = processed_raw_data.replace(-98., np.nan)

    return processed_raw_data


def change_data_types(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Change data types
    Args:
        raw_data:

    Returns:
        raw_data_with_none with changed data types
    """
    # Change data types
    raw_data = raw_data.convert_dtypes()

    # Change date format
    raw_data["IntDate"] = pd.to_datetime(raw_data["IntDate"])

    return raw_data


def set_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set column 'SRCID' as the index
    Args:
        df:
    Returns:
        df with column 'SRCID' as the index
    """
    # Set column 'SRCID' as the index
    df = df.set_index('SRCID')

    return df


def process_raw_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw data
    Args:
        raw_data:

    Returns:
        processed raw data
    """
    # Replace empty strings with None values
    processed_raw_data = replace_empty_strings_with_none(raw_data)

    # Change data types
    processed_raw_data = change_data_types(processed_raw_data)

    # Set column 'SRCID' as the index
    processed_raw_data = set_index(processed_raw_data)

    return processed_raw_data


def drop_redundant_columns(intermediate_data: pd.DataFrame) -> pd.DataFrame:
    """
    Drop redundant columns
    Args:
        intermediate_data:

    Returns:
        intermediate_data with dropped redundant columns
    """
    # Find the columns with only one unique value
    unique_counts = intermediate_data.nunique()
    only_one_unique = unique_counts[unique_counts == 1]

    # Get the list of column names
    cols_with_one_unique_value = list(only_one_unique.index)

    # Redundant columns
    redundant_cols = ['IntDate',
                      'Mode',
                      'CF_Order',
                      'DUM1',
                      'DUM2',
                      'Q5_Order',
                      'Q7_Order',
                      'Q9_Order',
                      'Q10_Order',
                      'Q11_Order',
                      'Q12_Order',
                      'SECTION_DUM',
                      'Q15_Order',
                      'Q16_Order',
                      'Q17_Order',
                      'wt_internet',
                      'wt_tel',
                      'wt_volunteer',
                      'wt_design',
                      'wt_propensit',
                      'wt_base_anu',
                      'wt_wave_anu',
                      'weight_anu',
                      ]

    intermediate_data = intermediate_data.drop(redundant_cols, axis=1)

    # Drop redundant columns
    intermediate_data = intermediate_data.drop(cols_with_one_unique_value, axis=1)

    return intermediate_data


def process_r_labels(codebook_xls: pd.ExcelFile) -> pd.DataFrame:
    """
    Processes the codebook to extract data from the attribute names and descriptions
    Args:
        codebook_xls:

    Returns:
        Data with information about the attributes
    """
    # Get the questionnaire description and labels from the codebook
    labels = codebook_xls[['Variable', 'Position', 'Label']]

    # Drop rows with no labels
    labels.dropna(inplace=True)

    # Converts the labels of questions R1-R31 into risk aversion values
    r_labels = labels.iloc[104:135]
    r_labels["ev_rf"] = r_labels["Label"].str.split(' ').str[33].str[1:].astype(int)
    r_labels["ev_gamble"] = 450
    r_labels["risk_premium"] = r_labels["ev_gamble"] - r_labels["ev_rf"]
    
    r_labels["eu_rf"] = r_labels["ev_rf"].apply(lambda x: x ** 0.5)
    r_labels["eu_gamble"] = math.sqrt(900) * 0.5
    r_labels["eu_diff"] = r_labels["eu_rf"] - r_labels["eu_gamble"]

    return r_labels


def process_s_labels(codebook_xls: pd.ExcelFile) -> pd.DataFrame:
    """
    Processes the codebook to extract data from the attribute names and descriptions
    Args:
        codebook_xls:

    Returns:
        Data with information about the attributes
    """
    # Get the questionnaire description and labels from the codebook
    labels = codebook_xls[['Variable', 'Position', 'Label']]

    # Drop rows with no labels
    labels.dropna(inplace=True)

    # Converts the labels of questions S1-S31 into annual return percentages
    s_labels = labels.iloc[136:167]
    s_labels["fv"] = s_labels["Label"].str.split(' ').str[8].str[1:].astype(int)
    s_labels["annual_return"] = (s_labels["fv"] - 300) / 300

    return s_labels


def define_min_risk_premium(processed_data: pd.DataFrame, r_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Define the minimum risk premium for each participant
    Args:
        processed_data:
        r_labels:

    Returns:
        feature_data with the minimum risk premium for each participant
    """

    r_cols = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11',
              'R12', 'R13', 'R14', 'R15', 'R16', 'R17', 'R18', 'R19', 'R20', 'R21',
              'R22', 'R23', 'R24', 'R25', 'R26', 'R27', 'R28', 'R29', 'R30', 'R31']

    df = processed_data[r_cols]

    risk_premium_labels = r_labels["risk_premium"].tolist()

    df = df.rename(columns=dict(zip(r_cols, risk_premium_labels)))

    df = df.replace(-99.0, np.nan)
    df = df.replace(-98.0, np.nan)

    # define a lambda function to apply to each row
    get_index = lambda row: row.index[row == 1.0].tolist()

    # apply the lambda function to each row of the DataFrame
    df["risk_premiums_to_bet"] = df.apply(get_index, axis=1)

    df["risk_premiums_to_bet"].fillna(0)

    # Define a function to return the minimum value of a list
    def get_min(lst):
        if len(lst) == 0:
            return None
        else:
            return min(lst)

    # Apply the function to each row of the DataFrame and assign the result to a new column
    df['min_risk_premium_to_bet'] = df["risk_premiums_to_bet"].apply(get_min)

    feature_data = df[['min_risk_premium_to_bet']]

    return feature_data


def define_min_return(primary_data: pd.DataFrame, s_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Define the minimum risk premium for each participant
    Args:
        primary_data:
        s_labels:

    Returns:
        feature_data with the minimum risk premium for each participant
    """

    s_cols = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11',
              'R12', 'R13', 'R14', 'R15', 'R16', 'R17', 'R18', 'R19', 'R20', 'R21',
              'R22', 'R23', 'R24', 'R25', 'R26', 'R27', 'R28', 'R29', 'R30', 'R31']

    df = primary_data[s_cols]

    annual_return_labels = s_labels["annual_return"].tolist()

    df = df.rename(columns=dict(zip(s_cols, annual_return_labels)))

    df = df.replace(-99.0, np.nan)
    df = df.replace(-98.0, np.nan)

    # define a lambda function to apply to each row
    get_index = lambda row: row.index[row == 2.0].tolist()

    # apply the lambda function to each row of the DataFrame
    df["annual_return"] = df.apply(get_index, axis=1)

    df["annual_return"].fillna(0)

    # Define a function to return the minimum value of a list
    def get_min(lst):
        if len(lst) == 0:
            return None
        else:
            return min(lst)

    # Apply the function to each row of the DataFrame and assign the result to a new column
    df['min_annual_return'] = df["annual_return"].apply(get_min)

    feature_data_s = df[['min_annual_return']]

    return feature_data_s


def generate_risk_aversion_features(feature_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate features for risk aversion
    Args:
        feature_data:

    Returns:
        Features for risk aversion
    """
    # 0.1-1 normalize 'Column1' where 1 is the minimum value and 0 is the maximum value
    feature_data['risk_premium_normalised'] = 0.1 + (
            feature_data['min_risk_premium_to_bet'].max() - feature_data['min_risk_premium_to_bet']) / (
                                                      feature_data['min_risk_premium_to_bet'].max() - feature_data[
                                                  'min_risk_premium_to_bet'].min()) * 0.9

    feature_data['min_risk_premium_to_bet'] = feature_data['min_risk_premium_to_bet'].fillna(0)
    feature_data['risk_premium_normalised'] = feature_data['risk_premium_normalised'].fillna(0)

    # create new column based on values of old column
    feature_data['risk_aversion'] = feature_data['min_risk_premium_to_bet'].apply(
        lambda x: 2 if x < 0 else (1 if x > 0 else 0))

    return feature_data


def select_target_variable(primary_data: pd.DataFrame) -> pd.DataFrame:
    """
    Select the target variable for the problem
    :param primary_data:
    :return:
    """
    primary_data = primary_data.replace(-99.0, np.nan)
    primary_data = primary_data.replace(-98.0, np.nan)

    gambling_problem_cols = ['Q15a',
                             'Q15b',
                             'Q15c',
                             'Q15d',
                             'Q15e',
                             'Q15f',
                             'Q15g',
                             'Q15h',
                             'Q15i', ]

    df = primary_data[gambling_problem_cols]

    # Create a new column that is the sum of all values in each row
    df['gambling_problem'] = df.sum(axis=1)

    df['gambling_problem'] = df['gambling_problem'].replace(0, np.nan)

    # Normalize the 'gambling_problem' column using the formula
    df['gambling_problem_normalised'] = (df['gambling_problem'].max() - df['gambling_problem']) / (
            df['gambling_problem'].max() - df['gambling_problem'].min())

    # Create a new column using a lambda function to apply the conditions
    df['gambling_problem_binary'] = df['gambling_problem_normalised'].apply(
        lambda x: 0 if x == 0 else None if pd.isna(x) else 1)

    return df


def generate_gambling_variety_feature(primary_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a feature for gambling variety
    :param primary_data:
    :return:
    """

    gambling_variety = primary_data[['Q13_1',
                                     'Q13_2',
                                     'Q13_3',
                                     'Q13_4',
                                     'Q13_5',
                                     'Q13_6',
                                     'Q13_7',
                                     'Q13_8',
                                     'Q13_9',
                                     'Q13_10',
                                     'Q13_11', ]]

    gambling_variety["gambling_variety"] = gambling_variety.sum(axis=1)

    return gambling_variety["gambling_variety"]


def select_other_features(primary_data: pd.DataFrame) -> pd.DataFrame:
    """
    Select other features
    :param primary_data:
    :return:
    """

    qts = ['Q16a',
           'Q16b',
           'Q16c',
           'Q16d',
           'Q16e',
           'Q16f',
           'Q16g',
           'Q16h',
           'Q16i',
           'Q17a',
           'Q17b',
           'Q17c',
           'Q17d',
           'Q18',
           'Q20',
           'Q22',
           ]

    df = primary_data[qts]

    return df


def generate_model_input(other_features: pd.DataFrame, min_return: pd.DataFrame, risk_aversion: pd.DataFrame,
                         gambling_variety: pd.DataFrame, target_variable: pd.DataFrame) -> pd.DataFrame:
    """
    :param
    other_features:
    min_return:
    risk_aversion:
    gambling_variety:
    target_variable:
    :return:
    """
    # Join the two dataframes on row index, keeping only common rows
    model_input = other_features.join(min_return, how='inner')
    model_input = model_input.join(risk_aversion, how='inner')
    model_input = model_input.join(gambling_variety, how='inner')
    model_input = model_input.join(target_variable, how='inner')

    # Drop rows with missing values
    model_input = model_input.dropna(axis=0)

    return model_input
