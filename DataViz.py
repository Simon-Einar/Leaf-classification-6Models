import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.decomposition import PCA

class ResultVisualizer:
    def __init__(self):
        """Initialize the ResultVisualizer class."""
        pass

    def display_best_scores(self, scores, model_list):
        """Visualize the best scores achieved by different models."""
        model_names = [model_info['model_name'] for model_info in model_list]
        score_data = pd.DataFrame({"Model": model_names, "Best Score": scores})

        plt.figure(figsize=(10, 6))
        bars = plt.bar(score_data['Model'], score_data['Best Score'], color='skyblue')
        plt.bar_label(bars, labels=[f"{score:.2%}" for score in score_data['Best Score']])

        plt.title('Top Scores Across Models')
        plt.xlabel('Model')
        plt.ylabel('Best Score')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def display_evaluation_metrics(self, metrics_results):
        """Plot evaluation metrics for all models."""
        metrics_df = pd.DataFrame(metrics_results).T  # Transpose for better readability

        for metric in metrics_df.columns:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(metrics_df.index, metrics_df[metric], color='teal')
            plt.bar_label(bars, labels=[f"{val:.2%}" for val in metrics_df[metric]])

            plt.title(f'Comparison of {metric.capitalize()} Across Models')
            plt.xlabel('Model')
            plt.ylabel(metric.capitalize())
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

    def show_learning_curves(self, X, y, classifiers, metric='accuracy'):
        
        for model_info in classifiers:
            model_instance = model_info['instance']
            train_sizes, train_scores, val_scores = learning_curve(
                estimator=model_instance, X=X, y=y, scoring=metric, n_jobs=-1, cv=5
            )

            avg_train_scores = train_scores.mean(axis=1)
            std_train_scores = train_scores.std(axis=1)
            avg_val_scores = val_scores.mean(axis=1)
            std_val_scores = val_scores.std(axis=1)

            plt.figure()
            plt.title(f"Learning Curve: {model_info['model_name']}")
            plt.xlabel("Number of Training Samples")
            plt.ylabel(metric)
            plt.grid()

            plt.fill_between(train_sizes, avg_train_scores - std_train_scores,
                            avg_train_scores + std_train_scores, alpha=0.1, color="blue")
            plt.fill_between(train_sizes, avg_val_scores - std_val_scores,
                            avg_val_scores + std_val_scores, alpha=0.1, color="orange")
            plt.plot(train_sizes, avg_train_scores, 'o-', color="blue", label="Training Score")
            plt.plot(train_sizes, avg_val_scores, 'o-', color="orange", label="Validation Score")

            plt.legend(loc="best")
            plt.show()



    def visualize_explained_variance(self, dataset):
        """Plot the explained variance to determine optimal components using PCA."""
        pca_model = PCA()
        pca_model.fit(dataset)
        cumulative_variance = np.cumsum(pca_model.explained_variance_ratio_)

        plt.figure(figsize=(8, 5))
        plt.plot(cumulative_variance, marker='o', linestyle='--')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Optimal Number of PCA Components')
        plt.grid()
        plt.tight_layout()
        plt.show()
