"""
This is a boilerplate pipeline 'data_processing_pipeline'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=process_raw_data,
            inputs="raw_data",
            outputs="intermediate_data",
            name="process_raw_data",
        ),
        node(func=drop_redundant_columns,
             inputs="intermediate_data",
             outputs="primary_data",
             name="drop_redundant_columns",
             ),
        node(func=process_r_labels,
             inputs="codebook",
             outputs="r_labels",
             name="process_r_labels",
             ),
        node(func=define_min_risk_premium,
             inputs=["primary_data", "r_labels"],
             outputs="feature_data_r",
             name="define_min_risk_premium",
             ),
        node(func=process_s_labels,
             inputs="codebook",
             outputs="s_labels",
             name="process_s_labels",
             ),
        node(func=define_min_return,
             inputs=["primary_data", "s_labels"],
             outputs="min_return",
             name="define_min_return",
             ),
        node(func=generate_risk_aversion_features,
             inputs="feature_data_r",
             outputs="risk_aversion",
             name="generate_risk_aversion_features",
             ),
        node(func=select_target_variable,
             inputs="primary_data",
             outputs="target_variable",
             name="select_target_variable"
             ),
        node(func=generate_gambling_variety_feature,
             inputs="primary_data",
             outputs="gambling_variety",
             name="generate_gambling_variety_feature"
             ),
        node(func=select_other_features,
             inputs="primary_data",
             outputs="other_features",
             name="select_other_features"
             ),
        node(func=generate_model_input,
             inputs=['other_features', 'min_return', 'risk_aversion', 'gambling_variety', 'target_variable'],
             outputs="model_input",
             name="generate_model_input"
             ),
    ])
