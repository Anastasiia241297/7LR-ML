from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        """
        Обучает новую базовую модель и добавляет ее в ансамбль.
        """
        # Вычисляем псевдо-ответы
        gradients = self.loss_derivative(y, predictions)

        # Обучаем новую базовую модель на подвыборке данных
        indices = np.random.choice(x.shape[0], int(x.shape[0] * self.subsample), replace=False)
        x_sampled, y_sampled = x[indices], gradients[indices]

        model = self.base_model_class(**self.base_model_params)
        model.fit(x_sampled, y_sampled)

        # Предсказания новой модели
        new_predictions = model.predict(x)

        # Находим оптимальную гамму
        gamma = self.find_optimal_gamma(y, predictions, new_predictions)

        # Сохраняем модель и гамму
        self.models.append(model)
        self.gammas.append(gamma)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        Обучает модель на тренировочном наборе данных и выполняет валидацию на валидационном наборе.

        Параметры
        ----------
        x_train : array-like, форма (n_samples, n_features)
            Массив признаков для тренировочного набора.
        y_train : array-like, форма (n_samples,)
            Массив целевых значений для тренировочного набора.
        x_valid : array-like, форма (n_samples, n_features)
            Массив признаков для валидационного набора.
        y_valid : array-like, форма (n_samples,)
            Массив целевых значений для валидационного набора.
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])
        best_loss = float('inf')
        rounds_no_improvement = 0

        for iteration in range(self.n_estimators):
            # Обучаем новую базовую модель
            self.fit_new_base_model(x_train, y_train, train_predictions)

            # Обновляем предсказания
            train_predictions += self.learning_rate * self.models[-1].predict(x_train) * self.gammas[-1]
            valid_predictions += self.learning_rate * self.models[-1].predict(x_valid) * self.gammas[-1]

            # Вычисляем ошибки
            train_loss = self.loss_fn(y_train, train_predictions)
            valid_loss = self.loss_fn(y_valid, valid_predictions)

            # Сохраняем историю ошибок
            self.history['train_loss'].append(train_loss)
            self.history['valid_loss'].append(valid_loss)

            # Проверка на раннюю остановку
            if self.early_stopping_rounds is not None:
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    rounds_no_improvement = 0
                else:
                    rounds_no_improvement += 1
                    if rounds_no_improvement >= self.early_stopping_rounds:
                        print(f"Ранняя остановка на итерации {iteration + 1}")
                        break

        # Построение графика, если флаг plot установлен
        if self.plot:
            import matplotlib.pyplot as plt
            plt.plot(self.history['train_loss'], label='Train Loss')
            plt.plot(self.history['valid_loss'], label='Validation Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training and Validation Loss')
            plt.show()

    def predict_proba(self, x):
        """
        Вычисляет вероятности принадлежности классу для каждого образца.
        """
        predictions = np.zeros(x.shape[0])
        for gamma, model in zip(self.gammas, self.models):
            predictions += self.learning_rate * gamma * model.predict(x)

        probabilities = self.sigmoid(predictions)
        return np.vstack((1 - probabilities, probabilities)).T

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        """
        Находит оптимальное значение гаммы для минимизации функции потерь.

        Параметры
        ----------
        y : array-like, форма (n_samples,)
            Целевые значения.
        old_predictions : array-like, форма (n_samples,)
            Предыдущие предсказания ансамбля.
        new_predictions : array-like, форма (n_samples,)
            Новые предсказания базовой модели.

        Возвращает
        ----------
        gamma : float
            Оптимальное значение гаммы.

        Примечания
        ----------
        Значение гаммы определяется путем минимизации функции потерь.
        """
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        """
        Возвращает важность признаков в обученной модели.

        Возвращает
        ----------
        importances : array-like, форма (n_features,)
            Важность каждого признака.

        Примечания
        ----------
        Важность признаков определяется по вкладу каждого признака в финальную модель.
        """
        pass
