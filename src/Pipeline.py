from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from src.Adapters import log_adapted

pipeline_Log_LR = Pipeline([
    ('log_transform', log_adapted),
    ('model', LinearRegression())
])
