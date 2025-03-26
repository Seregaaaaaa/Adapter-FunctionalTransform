import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CustomFunctionTransformer(BaseEstimator, TransformerMixin):
    """
    Конструирует трансформер из произвольной функции.
    
    Используется как адаптер для подключения пользовательских функций
    к конвейерам обработки данных scikit-learn.
    
    Parameters
    ----------
    func : callable, default=None
        Функция преобразования. Если None, возвращает исходный ввод.
    
    inverse_func : callable, default=None
        Обратная функция преобразования. Если None, обратное преобразование
        вызывает NotImplementedError.
    
    validate : bool, default=False
        Проверять ли входные данные X.
    
    accept_sparse : bool, default=False
        Принимать ли разреженные матрицы в качестве входных данных.
    
    check_inverse : bool, default=True
        Проверять ли согласованность func и inverse_func.
    
    kw_args : dict, default=None
        Аргументы для передачи в func.
    
    inv_kw_args : dict, default=None
        Аргументы для передачи в inverse_func.
    """
    
    def __init__(self, func=None, inverse_func=None, validate=False,
                 accept_sparse=False, check_inverse=True,
                 kw_args=None, inv_kw_args=None):
        self.func = func
        self.inverse_func = inverse_func
        self.validate = validate
        self.accept_sparse = accept_sparse
        self.check_inverse = check_inverse
        self.kw_args = kw_args
        self.inv_kw_args = inv_kw_args
    
    def _check_input(self, X):
        """Проверяет входные данные."""
        if self.validate:
            if self.accept_sparse:
                # Проверка для разреженных матриц
                if not (hasattr(X, "data") and hasattr(X, "indices") and 
                        hasattr(X, "indptr") and hasattr(X, "shape")):
                    if not isinstance(X, np.ndarray):
                        raise TypeError("X должен быть массивом numpy или разреженной матрицей")
            else:
                # Проверка только для numpy массивов
                if not isinstance(X, np.ndarray):
                    raise TypeError("X должен быть массивом numpy")
        return X
    
    def fit(self, X, y=None):
        """Не делает ничего: этап обучения отсутствует."""
        X = self._check_input(X)
        return self
    
    def transform(self, X):
        """
        Применяет функцию трансформации к X.
        
        Parameters
        ----------
        X : array-like или разреженная матрица
            Входные данные для преобразования.
            
        Returns
        -------
        X_out : array-like или разреженная матрица
            Преобразованные данные.
        """
        X = self._check_input(X)
        
        if self.func is None:
            return X
        
        args = {} if self.kw_args is None else self.kw_args
        return self.func(X, **args)
    
    def inverse_transform(self, X):
        """
        Применяет обратную функцию трансформации к X.
        
        Parameters
        ----------
        X : array-like или разреженная матрица
            Входные данные для обратного преобразования.
            
        Returns
        -------
        X_out : array-like или разреженная матрица
            Обратно преобразованные данные.
        """
        X = self._check_input(X)
        
        if self.inverse_func is None:
            raise NotImplementedError("Обратная функция не предоставлена.")
        
        args = {} if self.inv_kw_args is None else self.inv_kw_args
        return self.inverse_func(X, **args)
    
    def __sklearn_is_fitted__(self):
        """Сообщает scikit-learn, что объект всегда готов."""
        return True