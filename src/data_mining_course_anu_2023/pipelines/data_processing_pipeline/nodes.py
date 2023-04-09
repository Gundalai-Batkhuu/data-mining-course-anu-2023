from kedro.pipeline import *
from kedro.io import *
from kedro.runner import *
import pandas as pd
import numpy as np
import pickle
import os


def replace_empty_strings_with_none(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Replace empty strings with None values
    Args:
        raw_data:

    Returns:
        raw_data with empty strings replaced with None values
    """
    # Replace empty strings with None values
    raw_data.replace(' ', np.nan, inplace=True)

    return raw_data


def change_data_types(raw_data_with_none: pd.DataFrame) -> pd.DataFrame:
    """
    Change data types
    Args:
        raw_data_with_none:

    Returns:
        raw_data_with_none with changed data types
    """
    # Change data types
    raw_data_with_none = raw_data_with_none.convert_dtypes()

    # Change date format
    raw_data_with_none["IntDate"] = pd.to_datetime(raw_data_with_none["IntDate"])

    return raw_data_with_none


