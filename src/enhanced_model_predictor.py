import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.svm import SVR
from typing import Tuple, List, Dict, Any, Optional
import logging

# Try to import the neural network model
try:
    from neural_network_model import NeuralNetworkModel
    NN_AVAILABLE = True
except ImportError:
    NN_AVAILABLE = False
    logging.debug("Neural network model not available. Will use other models only.")

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

class EnhancedModelPredictor:
    """
    Enhanced ML model predictor with multiple algorithms and advanced feature engineering.
    This class reduces reliance on hardcoded priorities by using a more sophisticated
    ensemble of ML models with better feature engineering.
    """
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.model_weights = {}
        self.feature_cols = None
        self.categorical_cols = []
        self.numerical_cols = []
        self.advanced_features = []
        self.priority_weight = 0.75  # Reduced from 0.85 to 0.75 (25% reduction in priority influence)

    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced features for better prediction with more sophisticated transformations.
        """
        df = df.copy()
        
        # Reset index to avoid reindexing issues with duplicate indices
        df = df.reset_index(drop=True)
        
        # Convert all numeric columns to appropriate types first
        numeric_columns = ['Runs', 'Balls Faced', 'Fours', 'Sixes', 'Strike Rate', 'Overs',
                          'Runs Conceded', 'Wickets', 'Economy', 'Catches', 'Stumpings',
                          'Run Outs (Direct)', 'Run Outs (Assist)', 'Maidens', 'Runs Given']
        
        # Convert all columns in the dataframe to numeric, not just the predefined ones
        # This will handle any unexpected string values like 'LBW' in any column
        for col in df.columns:
            if col not in ['Player', 'Team', 'Match Number', 'Player Role', 'Venue', 'Opposition']:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Basic ratio features
        if 'Runs' in df.columns and 'Balls Faced' in df.columns and df['Balls Faced'].sum() > 0:
            df['Runs_per_Ball'] = df['Runs'] / df['Balls Faced'].replace(0, 1)
            df['Strike_Rate_Efficiency'] = df['Runs_per_Ball'] * 100 / 150  # Normalized to T20 benchmark
        
        if 'Wickets' in df.columns and 'Overs' in df.columns and df['Overs'].sum() > 0:
            df['Wickets_per_Over'] = df['Wickets'] / df['Overs'].replace(0, 1)
            df['Economy_Efficiency'] = 8 / df['Economy'].replace(0, 8)  # Normalized to T20 benchmark
        
        # Advanced batting metrics
        if all(col in df.columns for col in ['Runs', 'Fours', 'Sixes']):
            df['Boundary_Count'] = df['Fours'] + df['Sixes']
            df['Boundary_Percentage'] = df['Boundary_Count'] / df['Balls Faced'].replace(0, 1) * 100
            df['Non_Boundary_SR'] = (df['Runs'] - 4*df['Fours'] - 6*df['Sixes']) / (df['Balls Faced'] - df['Boundary_Count']).replace(0, 1) * 100
        
        # Advanced bowling metrics
        if all(col in df.columns for col in ['Wickets', 'Overs', 'Runs Given']):
            df['Bowling_Average'] = df['Runs Given'] / df['Wickets'].replace(0, 1)
            df['Bowling_SR'] = df['Overs'] * 6 / df['Wickets'].replace(0, 1)
            df['Dot_Ball_Estimate'] = df['Overs'] * 6 - df['Runs Given'] / 1.5  # Rough estimate
        
        # Role-specific features - completely rewritten to avoid reindexing issues
        if 'Player Role' in df.columns:
            # Instead of one-hot encoding, we'll create role-specific features directly
            # This avoids any potential reindexing or broadcasting issues
            
            # Initialize role-specific columns with zeros
            roles = ['BAT', 'BOWL', 'ALL', 'WK']
            
            # Create batting features for relevant roles
            if 'Runs' in df.columns:
                for role in ['BAT', 'ALL', 'WK']:
                    df[f'{role}_Runs'] = 0.0
                    df.loc[df['Player Role'].str.upper() == role, f'{role}_Runs'] = df.loc[df['Player Role'].str.upper() == role, 'Runs']
                    
                    if 'Strike Rate' in df.columns:
                        df[f'{role}_SR_Impact'] = 0.0
                        df.loc[df['Player Role'].str.upper() == role, f'{role}_SR_Impact'] = df.loc[df['Player Role'].str.upper() == role, 'Strike Rate'] / 100
            
            # Create bowling features for relevant roles
            if 'Wickets' in df.columns:
                for role in ['BOWL', 'ALL']:
                    df[f'{role}_Wickets'] = 0.0
                    df.loc[df['Player Role'].str.upper() == role, f'{role}_Wickets'] = df.loc[df['Player Role'].str.upper() == role, 'Wickets']
                    
                    if 'Economy' in df.columns:
                        df[f'{role}_Economy_Impact'] = 0.0
                        df.loc[df['Player Role'].str.upper() == role, f'{role}_Economy_Impact'] = 10 - df.loc[df['Player Role'].str.upper() == role, 'Economy']
        
        # Form and consistency metrics
        if 'Match Number' in df.columns and 'Total Points' in df.columns:
            df = df.sort_values(['Player', 'Match Number'])
            df['Recent_Form'] = df.groupby('Player')['Total Points'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
            df['Form_Trend'] = df.groupby('Player')['Total Points'].transform(lambda x: x.rolling(window=5, min_periods=1).apply(lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0))
            df['Consistency'] = df.groupby('Player')['Total Points'].transform(lambda x: 1 / (x.rolling(window=5, min_periods=1).std() + 1))
        
        # Venue and opposition features if available
        if 'Venue' in df.columns:
            venue_dummies = pd.get_dummies(df['Venue'], prefix='Venue')
            df = pd.concat([df, venue_dummies], axis=1)
        
        if 'Opposition' in df.columns:
            opposition_dummies = pd.get_dummies(df['Opposition'], prefix='Opposition')
            df = pd.concat([df, opposition_dummies], axis=1)
        
        # Store advanced feature names for later use
        self.advanced_features = [col for col in df.columns if col not in numeric_columns and 
                                col not in ['Player', 'Team', 'Match Number', 'Total Points', 'Player Role'] and
                                not col.startswith('Venue_') and not col.startswith('Opposition_')]
        
        return df

    def identify_feature_types(self, df: pd.DataFrame) -> None:
        """
        Identify numerical and categorical features for preprocessing.
        """
        # Identify categorical columns (excluding target and metadata)
        self.categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and 
                               col not in ['Player', 'Team', 'Match Number', 'Total Points']]
        
        # Identify numerical columns (excluding target and metadata)
        self.numerical_cols = [col for col in df.columns if df[col].dtype != 'object' and 
                              col not in ['Player', 'Team', 'Match Number', 'Total Points']]

    def train(self, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Dict[str, Any]:
        """
        Train an enhanced ensemble of models with feature engineering and model tuning.
        """
        df = df.copy()
        
        # Reset index to avoid reindexing issues with duplicate indices
        df = df.reset_index(drop=True)
        
        # Create advanced features
        df = self.create_advanced_features(df)
        
        # Identify feature types
        self.identify_feature_types(df)
        
        # Ensure all feature columns exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Add advanced features to feature columns
        all_features = feature_cols + self.advanced_features
        
        # Handle missing values and convert all features to numeric
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(0)
        
        # Convert all feature columns to numeric, handling non-numeric values
        for col in all_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        X = df[all_features]
        y = df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize models with optimized hyperparameters
        models = {
            'rf': RandomForestRegressor(
                n_estimators=200, 
                max_depth=10, 
                min_samples_split=5, 
                random_state=42
            ),
            'gbr': GradientBoostingRegressor(
                n_estimators=200, 
                learning_rate=0.05, 
                max_depth=6, 
                subsample=0.8, 
                random_state=42
            ),
            'ada': AdaBoostRegressor(
                n_estimators=100,
                learning_rate=0.05,
                random_state=42
            ),
            'en': ElasticNet(
                alpha=0.5,
                l1_ratio=0.5,
                random_state=42
            ),
            'svr': SVR(
                kernel='rbf',
                C=1.0,
                epsilon=0.1
            ),
            'mlp': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=500,
                random_state=42
            )
        }
        
        # Add optional models if available
        if XGBRegressor:
            models['xgb'] = XGBRegressor(
                n_estimators=200, 
                learning_rate=0.05, 
                max_depth=6, 
                subsample=0.8, 
                colsample_bytree=0.8, 
                random_state=42, 
                verbosity=0
            )
        
        if CatBoostRegressor:
            models['cat'] = CatBoostRegressor(
                iterations=200, 
                learning_rate=0.05, 
                depth=6, 
                subsample=0.8, 
                random_state=42, 
                verbose=0
            )
            
        if LGBMRegressor:
            models['lgbm'] = LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                random_state=42,
                verbose=-1
            )
            
        # Add neural network model if available
        if NN_AVAILABLE:
            try:
                nn_model = NeuralNetworkModel()
                nn_result = nn_model.train(df, all_features, target_col)
                if nn_result.get('trained', False):
                    self.models['nn'] = nn_model
                    # Use inverse MSE as weight
                    self.model_weights['nn'] = 1 / (nn_result.get('mse', 1e10) + 1e-10)
                    logging.info("Neural network model trained successfully.")
                else:
                    logging.warning(f"Neural network training failed: {nn_result.get('error', 'Unknown error')}")
            except Exception as e:
                logging.warning(f"Error training neural network model: {e}")
                pass
        
        # Train models and calculate weights based on performance using cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in models.items():
            try:
                # Train the model
                if name in ['en', 'svr', 'mlp']:
                    model.fit(X_train_scaled, y_train)
                    self.models[name] = model
                else:
                    model.fit(X_train, y_train)
                    self.models[name] = model
                
                # Calculate cross-validation score
                if name in ['en', 'svr', 'mlp']:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='neg_mean_squared_error')
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
                
                # Convert negative MSE to positive weight (higher is better)
                self.model_weights[name] = np.mean(-cv_scores)
                
                # Store feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(zip(all_features, model.feature_importances_))
            except Exception as e:
                logging.warning(f"Error training model {name}: {e}")
                continue
        
        # Invert weights (lower MSE = higher weight) and normalize to sum to 1
        for name in self.model_weights:
            self.model_weights[name] = 1 / (self.model_weights[name] + 1e-10)  # Add small epsilon to avoid division by zero
        
        total_weight = sum(self.model_weights.values())
        self.model_weights = {k: v/total_weight for k, v in self.model_weights.items()}
        
        self.scaler = scaler
        self.feature_cols = all_features
        
        # Calculate ensemble performance
        ensemble_pred = self.predict(df.head(len(X_test)))[target_col].values
        ensemble_score = r2_score(y_test, ensemble_pred)
        
        return {
            'models': self.models,
            'ensemble_score': ensemble_score,
            'feature_importance': self.feature_importance,
            'model_weights': self.model_weights
        }

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions using the weighted ensemble of models.
        """
        df = df.copy()
        
        # Reset index to avoid reindexing issues with duplicate indices
        df = df.reset_index(drop=True)
        
        # Create advanced features
        df = self.create_advanced_features(df)
        
        # Ensure all feature columns exist
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Handle missing values and convert all features to numeric
        for col in self.feature_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        X = df[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        
        # Make weighted predictions
        weighted_preds = np.zeros(X.shape[0])
        model_preds = {}
        
        for name, model in self.models.items():
            try:
                if name == 'nn':
                    # Neural network model has its own predict method
                    pred = model.predict(df[self.feature_cols])
                elif name in ['en', 'svr', 'mlp']:
                    pred = model.predict(X_scaled)
                else:
                    pred = model.predict(X)
                
                # Store individual model predictions
                model_preds[name] = pred
                
                # Apply model weight
                weighted_preds += pred * self.model_weights[name]
            except Exception as e:
                logging.warning(f"Error predicting with model {name}: {e}")
                continue
        
        # Store predictions in DataFrame
        df['selection_score'] = weighted_preds
        
        # Store individual model predictions for analysis
        for name, preds in model_preds.items():
            df[f'model_{name}_score'] = preds
        
        return df

    def get_feature_importance_analysis(self) -> Dict:
        """
        Get aggregated feature importance across all models.
        """
        if not self.feature_importance:
            return {}
            
        # Average feature importance across models
        all_features = set()
        for importance in self.feature_importance.values():
            all_features.update(importance.keys())
            
        avg_importance = {}
        for feature in all_features:
            importances = []
            for model_imp in self.feature_importance.values():
                if feature in model_imp:
                    importances.append(model_imp[feature])
            avg_importance[feature] = np.mean(importances) if importances else 0
            
        return avg_importance

    def get_model_weights(self) -> Dict[str, float]:
        """
        Get the weights assigned to each model in the ensemble.
        """
        return self.model_weights

    def get_priority_weight(self) -> float:
        """
        Get the weight assigned to priority in the final selection score.
        """
        return self.priority_weight