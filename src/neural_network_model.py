import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, List, Tuple, Any
import logging

try:
    import tensorflow as tf
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.debug("TensorFlow not available. Neural network model will not be used.")

class NeuralNetworkModel:
    """
    Neural Network model for player performance prediction.
    This model can be used as part of the ensemble approach.
    """
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.is_trained = False

    def build_model(self, input_dim: int) -> Any:
        """
        Build a neural network model for regression.
        """
        if not TF_AVAILABLE:
            logging.warning("TensorFlow not available. Cannot build neural network model.")
            return None

        model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def train(self, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Dict[str, Any]:
        """
        Train the neural network model.
        """
        if not TF_AVAILABLE:
            logging.warning("TensorFlow not available. Cannot train neural network model.")
            return {'trained': False, 'error': 'TensorFlow not available'}

        # Ensure all feature columns exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        # Handle missing values
        df[target_col] = df[target_col].fillna(0)
        df[feature_cols] = df[feature_cols].fillna(0)

        X = df[feature_cols].values
        y = df[target_col].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Build model
        self.model = self.build_model(X_train_scaled.shape[1])
        if self.model is None:
            return {'trained': False, 'error': 'Failed to build model'}

        # Train model with early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        try:
            history = self.model.fit(
                X_train_scaled, y_train,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )

            # Evaluate model
            y_pred = self.model.predict(X_test_scaled, verbose=0).flatten()
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            self.feature_cols = feature_cols
            self.is_trained = True

            return {
                'trained': True,
                'mse': mse,
                'r2': r2,
                'history': history.history
            }

        except Exception as e:
            logging.error(f"Error training neural network model: {e}")
            return {'trained': False, 'error': str(e)}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained neural network model.
        """
        if not self.is_trained or self.model is None:
            logging.warning("Neural network model not trained. Cannot make predictions.")
            return np.zeros(len(df))

        # Ensure all feature columns exist
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0

        # Handle missing values
        df[self.feature_cols] = df[self.feature_cols].fillna(0)

        X = df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)

        try:
            predictions = self.model.predict(X_scaled, verbose=0).flatten()
            return predictions
        except Exception as e:
            logging.error(f"Error making predictions with neural network model: {e}")
            return np.zeros(len(df))

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance using permutation importance.
        Note: Neural networks don't have built-in feature importance,
        so this is a placeholder for compatibility with other models.
        """
        return {}