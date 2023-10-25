import numpy as np

class ModelEvaluator:
    def rmsle(y_real, y_pred):
        log1 = np.log(y_real+1)
        log2 = np.log(y_pred+1)
        calc = (log1 - log2) ** 2
        return np.sqrt(np.mean(calc))