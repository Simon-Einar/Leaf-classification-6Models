from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.utils import all_estimators
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class ModelTrainer:
    def __init__(self):
        self.all_models = all_estimators(type_filter="classifier")

    def match_classifiers(self, model_list):
        """Retrieve and initialize the classifiers based on user input."""
        matched_models = []
        for model_info in model_list:
            for name, model_class in self.all_models:
                if name == model_info["model_name"]:
                    model_instance = model_class()
                    model_info["instance"] = model_instance
                    matched_models.append(model_info)
        return matched_models

    def fit_models(self, X_train, y_train, model_configs, scoring_metric="accuracy", cv_folds=5):
        """Perform grid search to optimize the models and return trained instances."""
        trained_models = []
        optimal_estimators = []
        best_performance_scores = []

        for model_config in model_configs:
            grid_search = GridSearchCV(
                estimator=model_config["instance"],
                param_grid=model_config["hyperparameters"],
                scoring=scoring_metric,
                cv=cv_folds,
            )
            grid_search.fit(X_train, y_train)

            trained_models.append(grid_search)
            best_performance_scores.append(grid_search.best_score_)
            optimal_estimators.append(grid_search.best_estimator_)

        return trained_models, best_performance_scores, optimal_estimators

    def evaluate_metrics(self, true_labels, predictions):
        """Calculate evaluation metrics for a model."""
        evaluation_scores = {
            "accuracy": accuracy_score(true_labels, predictions),
            "recall": recall_score(true_labels, predictions, average="weighted"),
            "precision": precision_score(true_labels, predictions, average="weighted"),
            "f1_score": f1_score(true_labels, predictions, average="weighted"),
        }
        return evaluation_scores

    def assess_models(self, X_test, y_test, trained_models, model_list):
        """Generate predictions and evaluate models."""
        results_summary = {}
        for model_config, trained_model in zip(model_list, trained_models):
            model_name = model_config["model_name"]
            predictions = trained_model.predict(X_test)
            metrics = self.evaluate_metrics(y_test, predictions)
            results_summary[model_name] = metrics

        return results_summary
