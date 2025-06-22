import json
import pandas as pd
from datasets import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import xgboost as xgb
import wandb
from typing import Dict, Any, List, Tuple
import os


class NowcastingModel:
    """Solar irradiance nowcasting using XGBoost."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.training_config = config['training']
        self.nowcasting_config = config['nowcasting']
        
        # Initialize wandb
        wandb.login(key=self.training_config['wandb_key'])
        
        self.study_name = self.nowcasting_config['study_name']
        
        # Normalization parameters (will be set during training)
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None

    def load_data(self, dataset_name: str, embeddings_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load dataset and embeddings."""
        print(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split="train")
        df = pd.DataFrame(dataset)
        
        print(f"Loading embeddings: {embeddings_file}")
        embeddings = []
        with open(embeddings_file, "r") as file:
            for line in file:
                entry = json.loads(line)
                embeddings.append(entry["embedding"])
        
        embeddings = np.array(embeddings)
        
        # Prepare auxiliary features
        auxiliary_features = []
        targets = []
        
        for index in range(len(df)):
            row = df.iloc[index]
            
            # Solar geometry features
            zenith_angle = row["Zenith_angle"]
            azimuth_angle = row["Azimuth_angle"]
            panel_tilt = row["physics_panel_tilt"]
            panel_orientation = row["physics_panel_orientation"]
            aoi = row["physics_aoi"]
            
            # Create feature vector with trigonometric transformations
            features = [
                row["Clear_sky_ghi"],
                zenith_angle,
                azimuth_angle,
                panel_tilt,
                panel_orientation,
                aoi,
                row["physics_total_irradiance_tilted"],
                # Trigonometric features for better angle representation
                np.cos(np.deg2rad(zenith_angle)),
                np.sin(np.deg2rad(zenith_angle)),
                np.cos(np.deg2rad(azimuth_angle)),
                np.sin(np.deg2rad(azimuth_angle)),
                np.cos(np.deg2rad(panel_tilt)),
                np.sin(np.deg2rad(panel_tilt)),
                np.cos(np.deg2rad(panel_orientation)),
                np.sin(np.deg2rad(panel_orientation)),
                np.cos(np.deg2rad(aoi)),
                np.sin(np.deg2rad(aoi)),
            ]
            
            auxiliary_features.append(features)
            targets.append(row["Global_horizontal_irradiance"])
        
        auxiliary_features = np.array(auxiliary_features)
        targets = np.array(targets).reshape(-1, 1)
        
        return embeddings, auxiliary_features, targets

    def normalize_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """Normalize features."""
        if fit:
            self.X_mean = np.mean(X, axis=0)
            self.X_std = np.std(X, axis=0)
        
        return (X - self.X_mean) / self.X_std

    def normalize_targets(self, y: np.ndarray, fit: bool = True) -> np.ndarray:
        """Normalize targets."""
        if fit:
            self.y_mean = np.mean(y)
            self.y_std = np.std(y)
        
        return (y - self.y_mean) / self.y_std

    def denormalize_predictions(self, y_pred: np.ndarray) -> np.ndarray:
        """Denormalize predictions."""
        return y_pred * self.y_std + self.y_mean

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        # Denormalize predictions for evaluation
        y_true_denorm = self.denormalize_predictions(y_true)
        y_pred_denorm = self.denormalize_predictions(y_pred)
        
        mae = mean_absolute_error(y_true_denorm, y_pred_denorm)
        mse = mean_squared_error(y_true_denorm, y_pred_denorm)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_denorm, y_pred_denorm)
        nmap = mae / np.mean(y_true_denorm) * 100
        
        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2_Score": r2,
            "nMAP": nmap,
        }

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, params: Dict[str, Any]) -> Tuple[xgb.XGBRegressor, Dict[str, float]]:
        """Train XGBoost model."""
        model = xgb.XGBRegressor(**params)
        model.set_params(
            objective='reg:squarederror',
            eval_metric='mae',
            early_stopping_rounds=200
        )
        
        # Split for validation
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Train with early stopping
        evals = [(X_train_split, y_train_split), (X_val_split, y_val_split)]
        model.fit(
            X_train_split,
            y_train_split,
            eval_set=evals,
            verbose=False,
        )
        
        # Validate
        y_val_pred = model.predict(X_val_split)
        val_metrics = self.evaluate_model(y_val_split, y_val_pred)
        
        return model, val_metrics

    def objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """Optuna objective function."""
        wandb.init(
            project=self.study_name,
            mode="online",
            name=f"trial_{trial.number}",
            settings=wandb.Settings(init_timeout=150)
        )
        
        # Suggest hyperparameters
        params = {
            "booster": "gbtree",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 8),
            "lambda": trial.suggest_float("lambda", 0, 8),
        }
        
        # Train model
        model, val_metrics = self.train_model(X_train, y_train, params)
        
        # Log metrics
        wandb.log({"val_" + k: v for k, v in val_metrics.items()})
        wandb.log(params)
        
        wandb.finish()
        
        return val_metrics["nMAP"]

    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, n_trials: int = 100) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=10, n_warmup_steps=10, interval_steps=1
        )
        
        study = optuna.create_study(
            direction="minimize",
            pruner=pruner,
            study_name=self.study_name,
            storage=f"sqlite:///./{self.study_name}.db",
            load_if_exists=True,
        )
        
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        return study.best_trial.params

    def train_final_model(self, dataset_name: str, embeddings_file: str, model_save_path: str):
        """Train the final nowcasting model."""
        print("Loading data...")
        embeddings, auxiliary_features, targets = self.load_data(dataset_name, embeddings_file)
        
        # Combine features
        X = np.concatenate([embeddings, auxiliary_features], axis=1)
        y = targets
        
        # Normalize features and targets
        X_normalized = self.normalize_features(X, fit=True)
        y_normalized = self.normalize_targets(y, fit=True)
        
        print("Optimizing hyperparameters...")
        best_params = self.optimize_hyperparameters(
            X_normalized, y_normalized, 
            n_trials=self.nowcasting_config.get('n_trials', 100)
        )
        
        print(f"Best parameters: {best_params}")
        
        # Train final model with best parameters
        print("Training final model...")
        final_model = xgb.XGBRegressor(**best_params)
        final_model.set_params(objective='reg:squarederror', eval_metric='mae')
        final_model.fit(X_normalized, y_normalized.ravel())
        
        # Save model
        final_model.save_model(model_save_path)
        print(f"Model saved to {model_save_path}")
        
        # Save normalization parameters
        norm_params = {
            'X_mean': self.X_mean.tolist(),
            'X_std': self.X_std.tolist(),
            'y_mean': float(self.y_mean),
            'y_std': float(self.y_std)
        }
        
        norm_path = model_save_path.replace('.json', '_normalization.json')
        with open(norm_path, 'w') as f:
            json.dump(norm_params, f)
        
        print(f"Normalization parameters saved to {norm_path}")
        
        return final_model, best_params