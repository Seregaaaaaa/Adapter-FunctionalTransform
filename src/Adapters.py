import numpy as np
from src.CustomFunctionalTransformer import CustomFunctionTransformer

log_adapted = CustomFunctionTransformer(
    func=np.log,           
    inverse_func=np.exp,
    validate=True   
)