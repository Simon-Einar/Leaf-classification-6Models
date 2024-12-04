import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

class DataProcessor:
    def __init__(self):
        self.dataset = None

    def scale_features(self, data_frame, feature_columns):
        """Applies MinMax scaling to the specified columns."""
        scaler = MinMaxScaler()
        scaled_data = pd.DataFrame(scaler.fit_transform(data_frame), columns=feature_columns)
        return scaled_data

    def encode_labels(self, category_labels):
        """Converts categorical labels to numerical format."""
        label_encoder = LabelEncoder()
        label_encoder.fit(category_labels)
        return label_encoder.transform(category_labels)

    def split_dataset(self, data_frame, target_column):
        """Splits the dataset into training and testing sets."""
        features = data_frame.drop(columns=[target_column])
        target = data_frame[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.33, random_state=42, stratify=target
        )
        return X_train, X_test, y_train, y_test

    def apply_pca(self, dataset, num_components):

        # Initialize the PCA model
        pca_model = PCA(n_components=num_components)
        
        # Apply PCA transformation
        pca_result = pca_model.fit_transform(dataset)
        
        # Create a DataFrame for the principal components
        pca_columns = [f"principal_component_{i+1}" for i in range(num_components)]
        reduced_df = pd.DataFrame(pca_result, columns=pca_columns)
        
        # Add the original 'species' column to the result (assuming it's part of the dataset)
        if 'species' in dataset.columns:
            reduced_df = pd.concat([reduced_df, dataset['species']], axis=1)

        return reduced_df
