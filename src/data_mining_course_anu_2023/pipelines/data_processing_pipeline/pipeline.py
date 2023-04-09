"""
This is a boilerplate pipeline 'data_processing_pipeline'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=replace_empty_strings_with_none,
            inputs="raw_data",
            outputs="raw_data_with_none",
            name="replace_empty_strings_with_none",
        ),
        node(
            func=change_data_types,
            inputs="raw_data_with_none",
            outputs="intermediate_data",
            name="change_data_types",
        )
    ])
