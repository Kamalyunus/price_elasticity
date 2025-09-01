"""
Semi-Parametric Price Elasticity Model - Correct Implementation from Paper
Based on "Markdowns in E-Commerce Fresh Retail: A Counterfactual Prediction and Multi-Period Optimization Approach"

Key Equations Implemented:
1. Double-log parametric model (Equation 4):
   g(d_i; L_i, θ) = (θ_1 + θ_2^T L_i) ln(d_i) + c
   where d_i is discount ratio (1 - discount_pct), L_i is category encoding matrix

2. Parametric model fitting via least squares minimization:
   min ||ln(Y_i) - θ^T L̂_i ln(d_i) - c||² + λ||θ||²
   where L̂_i = [1, L_i^T] for parameters [θ_1, θ_2^T]

3. Base demand prediction using LGBM:
   h(d_i^o, x_i) = LGBM(features) → ln(Y_i)
   Y_i^o = exp(h(d_i^o, x_i))

4. Counterfactual prediction (Equation 8):
   ln Y_i,t(d_i) = ln Y_i,t^o + [g(d_i; L_i, θ) - g(d_i^o; L_i, θ)]
   = ln Y_i,t^o + θ^T L̂_i (ln d_i - ln d_i^o)
   
5. Final demand prediction:
   Y_i,t(d_i) = exp(ln Y_i,t(d_i))
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple, Optional, List
import yaml
import pickle
import warnings
warnings.filterwarnings('ignore')

def aggregate_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily data to weekly level to reduce noise in training
    
    Args:
        df: Daily dataframe with columns: date, sku, category, demand, discount_pct, base_price, etc.
    
    Returns:
        Weekly aggregated dataframe
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Add week identifiers
    df['year_week'] = df['date'].dt.strftime('%Y-%W')
    df['week_start'] = df['date'] - pd.to_timedelta(df['date'].dt.dayofweek, unit='d')
    
    # Aggregation rules
    agg_rules = {
        'demand': 'sum',  # Sum weekly demand
        'base_price': 'mean',  # Average base price for the week
        'discount_pct': 'mean',  # Average discount for the week
        'category': 'first',  # Category doesn't change
        'date': 'first',  # Keep first date of week for reference
    }
    
    # Add promotional features if they exist
    for col in df.columns:
        if col.startswith('promo_'):
            agg_rules[col] = 'max'  # If any promo during week, mark as 1
    
    # Calendar features - take mode or mean
    calendar_features = ['is_weekend', 'is_holiday', 'near_holiday']
    for feat in calendar_features:
        if feat in df.columns:
            agg_rules[feat] = 'mean'  # Proportion of week with this feature
    
    # Group by SKU and week
    df_weekly = df.groupby(['sku', 'year_week', 'week_start']).agg(agg_rules).reset_index()
    
    # Recreate calendar features at weekly level
    df_weekly['week_of_year'] = pd.to_datetime(df_weekly['week_start']).dt.isocalendar().week
    df_weekly['month'] = pd.to_datetime(df_weekly['week_start']).dt.month
    df_weekly['quarter'] = pd.to_datetime(df_weekly['week_start']).dt.quarter
    
    # Sort by date and SKU
    df_weekly = df_weekly.sort_values(['sku', 'week_start'])
    
    # Rename week_start to date for consistency
    df_weekly['date'] = df_weekly['week_start']
    df_weekly = df_weekly.drop(['year_week', 'week_start'], axis=1)
    
    return df_weekly

class DoubleLogParametricModel:
    """Double-log parametric model for price elasticity with hierarchical structure"""
    
    def __init__(self, config: Dict):
        self.config = config['model']['parametric']
        self.theta = None  # [θ_1, θ_2^T] parameters
        self.c = 0  # intercept
        self.fitted = False
        self.category_encoders = {}
        
    def encode_category_hierarchy(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create L_i matrix with hierarchical category encoding
        L_i = [category_1_onehot, category_2_onehot, category_3_onehot]
        """
        # Use the saved number of categories from training
        if not hasattr(self, 'n_categories'):
            # During training, set the number of categories
            categories = df['category'].unique()
            self.n_categories = len(categories)
            self.category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
        
        # Create L_i matrix with fixed dimensions from training
        L_matrix = np.zeros((len(df), self.n_categories))
        
        for i, (idx, row) in enumerate(df.iterrows()):
            category = row['category']
            if category in self.category_to_idx:
                cat_idx = self.category_to_idx[category]
                L_matrix[i, cat_idx] = 1
        
        return L_matrix
    
    def fit(self, df: pd.DataFrame):
        """
        Fit double-log parametric model using Equation (4) from paper:
        g(d_i; L_i, θ) = (θ_1 + θ_2^T L_i) ln(d_i) + c
        
        We fit this by minimizing: ||ln(Y_i/Y_i^nor) - θ^T L̂_i ln(d_i) - c||²
        """
        # Prepare data
        df_work = df.copy()
        
        # Use ln(Y_i) directly since we don't have separate normal channel
        # Add small constant to avoid log(0) issues
        df_work['ln_Y'] = np.log(df_work['demand'] + 1)  # ln(Y_i)
        
        # Calculate discount ratio (paper uses percentage discount d ∈ [0,1])
        # Convert discount percentage to discount ratio for log transformation
        df_work['discount_ratio'] = 1 - df_work['discount_pct']  # If 20% discount, ratio = 0.8
        df_work['ln_discount'] = np.log(df_work['discount_ratio'] + 1e-8)
        
        # Create hierarchical category matrix L_i
        L_matrix = self.encode_category_hierarchy(df_work)  # Shape: (n_samples, n_categories)
        
        # Create augmented L̂_i = [1, L_i^T] for [θ_1, θ_2^T]
        n_samples, n_categories = L_matrix.shape
        L_hat = np.column_stack([np.ones(n_samples), L_matrix])  # Shape: (n_samples, 1 + n_categories)
        
        # Create design matrix: each row is L̂_i * ln(d_i)
        ln_discounts = df_work['ln_discount'].values.reshape(-1, 1)
        X = L_hat * ln_discounts  # Element-wise multiplication, broadcasts correctly
        
        # Target variable
        y = df_work['ln_Y'].values
        
        # Solve least squares with regularization: min ||y - X*θ - c||² + λ||θ||²
        reg_strength = self.config.get('regularization_strength', 0.1)
        
        # Add intercept column
        X_with_intercept = np.column_stack([X, np.ones(n_samples)])
        
        # Solve regularized least squares
        XtX = X_with_intercept.T @ X_with_intercept
        # Add regularization (don't regularize intercept)
        reg_matrix = np.eye(XtX.shape[0]) * reg_strength
        reg_matrix[-1, -1] = 0  # Don't regularize intercept
        XtX += reg_matrix
        
        Xty = X_with_intercept.T @ y
        
        try:
            params = np.linalg.solve(XtX, Xty)
            self.theta = params[:-1]  # All except intercept
            self.c = params[-1]       # Intercept
        except np.linalg.LinAlgError:
            # Fallback to default values
            print("Warning: Could not solve least squares, using default values")
            self.theta = np.array([-1.5] + [-0.5] * n_categories)
            self.c = 0
        
        self.fitted = True
        
        # Calculate and display elasticities
        self._calculate_elasticities(df_work)
    
    def _calculate_elasticities(self, df: pd.DataFrame):
        """Calculate and display price elasticities by category"""
        if not self.fitted:
            return
        
        self.elasticities = {}
        
        # θ_1 is the base elasticity
        base_elasticity = self.theta[0]
        
        # For each category, elasticity = θ_1 + θ_2_k (where θ_2_k is category k's coefficient)
        for category, cat_idx in self.category_to_idx.items():
            category_coef = self.theta[1 + cat_idx] if len(self.theta) > 1 + cat_idx else 0
            elasticity = base_elasticity + category_coef
            self.elasticities[category] = elasticity
            
        print(f"Base elasticity (θ_1): {base_elasticity:.3f}")
        print("Category elasticities:")
        for cat, elast in self.elasticities.items():
            print(f"  {cat}: {elast:.3f}")
    
    def predict_log_ratio(self, df: pd.DataFrame, discount_ratios: np.ndarray) -> np.ndarray:
        """
        Predict elasticity effect using double-log model
        g(d_i; L_i, θ) = (θ_1 + θ_2^T L_i) ln(d_i) + c
        
        For baseline prediction (discount_ratio = 1.0), this returns the intercept c
        For other discounts, this returns the elasticity-adjusted effect
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        # Create L_i matrix for prediction data
        L_matrix = self.encode_category_hierarchy(df)
        
        # Create L̂_i = [1, L_i^T]
        n_samples, n_categories = L_matrix.shape
        L_hat = np.column_stack([np.ones(n_samples), L_matrix])
        
        # Calculate ln(discount_ratio)
        ln_discounts = np.log(discount_ratios + 1e-8)
        
        predictions = np.zeros(n_samples)
        
        for i in range(n_samples):
            # g(d_i; L_i, θ) = θ^T L̂_i * ln(d_i) + c
            theta_L = np.dot(self.theta, L_hat[i])  # (θ_1 + θ_2^T L_i)
            predictions[i] = theta_L * ln_discounts[i] + self.c
        
        return predictions
    
    def get_elasticity(self, category: str) -> float:
        """Get price elasticity for a specific category"""
        if not self.fitted or category not in self.elasticities:
            return -1.5  # Default elasticity
        return self.elasticities[category]

class BaseDemandLGBMModel:
    """LGBM model for base demand forecasting at base price/discount"""
    
    def __init__(self, config: Dict):
        self.config = config['model']['non_parametric']
        self.lgbm_params = self.config['lgbm_params']
        self.model = None
        self.feature_importance = None
        
    def prepare_features(self, df: pd.DataFrame, training: bool = False) -> pd.DataFrame:
        """Prepare features for base demand prediction at weekly level"""
        X = df.copy()
        
        # Use base price and discount percentage (as given in data)
        X['log_base_price'] = np.log(X['base_price'] + 1)
        
        # Calendar features (adjusted for weekly data)
        calendar_cols = ['week_of_year', 'month', 'quarter']
        if 'is_weekend' in X.columns:
            calendar_cols.append('is_weekend')  # Proportion of weekend days in week
        if 'is_holiday' in X.columns:
            calendar_cols.append('is_holiday')  # Proportion of holidays in week
        if 'near_holiday' in X.columns:
            calendar_cols.append('near_holiday')  # Proportion of near-holiday days
        
        # Promotional features
        promo_cols = [col for col in X.columns if col.startswith('promo_')]
        
        # Lag features (adjusted for weekly data - lag in weeks not days)
        lag_weeks = [1, 2, 4, 8]  # 1 week, 2 weeks, 4 weeks, 8 weeks ago
        for lag in lag_weeks:
            X[f'demand_lag_{lag}w'] = X.groupby('sku')['demand'].shift(lag)
            X[f'base_price_lag_{lag}w'] = X.groupby('sku')['base_price'].shift(lag)
            X[f'discount_lag_{lag}w'] = X.groupby('sku')['discount_pct'].shift(lag)
        
        # Rolling statistics (in weeks)
        for window in [4, 8]:  # 4-week and 8-week rolling averages
            X[f'demand_rolling_mean_{window}w'] = (
                X.groupby('sku')['demand'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
            )
            X[f'demand_rolling_std_{window}w'] = (
                X.groupby('sku')['demand'].transform(
                    lambda x: x.rolling(window, min_periods=2).std()
                )
            )
        
        # Year-over-year features if we have enough history
        X['demand_lag_52w'] = X.groupby('sku')['demand'].shift(52)  # Same week last year
        
        # Fill NaN values
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(0)
        
        # Include discount_pct - LGBM should see historical discount patterns
        feature_cols = (
            ['log_base_price', 'base_price', 'discount_pct'] +
            calendar_cols + promo_cols + 
            [col for col in X.columns if 'lag_' in col or 'rolling_' in col]
        )
        
        # Keep only existing columns
        feature_cols = [col for col in feature_cols if col in X.columns]
        
        return X[feature_cols]
    
    def fit(self, df: pd.DataFrame, df_val: Optional[pd.DataFrame] = None):
        """
        Fit LGBM for base demand prediction
        According to paper equation (2): E[ln(Y_i/Y_i^nor)|d_i^o] = h(d_i^o, x_i)
        """
        X_train = self.prepare_features(df, training=True)
        
        # Since we don't have separate normal channel, use ln(Y_i) directly
        # LGBM will predict ln(demand) at base discount level
        y_train = np.log(df['demand'].values + 1)  # h(d_i^o, x_i) = ln(Y_i)
        
        train_data = lgb.Dataset(X_train, label=y_train)
        
        if df_val is not None:
            X_val = self.prepare_features(df_val, training=False)
            y_val = np.log(df_val['demand'].values + 1)  # ln(Y_i)
            
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            self.model = lgb.train(
                self.lgbm_params,
                train_data,
                num_boost_round=self.config['n_estimators'],
                valid_sets=[valid_data],
                callbacks=[
                    lgb.early_stopping(self.config['early_stopping_rounds']),
                    lgb.log_evaluation(10)
                ]
            )
        else:
            self.model = lgb.train(
                self.lgbm_params,
                train_data,
                num_boost_round=self.config['n_estimators']
            )
        
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        print(f"Top 5 base demand features: {list(self.feature_importance.head()['feature'])}")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict base demand Y_i,t^o
        
        LGBM predicts h(d_i^o, x_i) = ln(Y_i)
        To get actual base demand: Y_i,t^o = exp(ln(Y_i))
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        X = self.prepare_features(df, training=False)
        log_demand_pred = self.model.predict(X, num_iteration=self.model.best_iteration)
        
        # Convert back to actual demand: Y_i = exp(ln(Y_i))
        actual_demand = np.exp(log_demand_pred)
        
        return actual_demand

class SemiParametricModel:
    """
    Correct Semi-Parametric Model Implementation from Paper
    
    Approach:
    1. LGBM predicts base demand at historical discount levels
    2. Double-log parametric model provides elasticity adjustment for counterfactual discounts
    """
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.base_demand_model = BaseDemandLGBMModel(self.config)
        self.parametric_model = DoubleLogParametricModel(self.config)
        self.fitted = False
        self.sku_historical_discounts = {}  # Store historical discount patterns per SKU
        
    def fit(self, df: pd.DataFrame):
        """Fit both components on weekly aggregated data
        
        Args:
            df: Input dataframe (will be aggregated to weekly if not already)
        """
        print("=" * 60)
        print("Training Semi-Parametric Model (Weekly Level)")
        print("=" * 60)
        
        # Always aggregate to weekly level
        if 'week_of_year' not in df.columns:
            print("\nAggregating data to weekly level...")
            df = aggregate_to_weekly(df)
            print(f"Data aggregated: {len(df)} weekly observations")
        
        # Store historical discount patterns per SKU for reference prediction
        self._store_historical_discount_patterns(df)
        
        # Split data in weeks
        test_weeks = self.config['training']['test_weeks']
        val_weeks = self.config['training']['validation_weeks']
        
        train_size = len(df) - test_weeks - val_weeks
        val_size = val_weeks
        
        df_train = df.iloc[:train_size]
        df_val = df.iloc[train_size:train_size+val_size] if val_size > 0 else None
        
        # 1. Fit base demand model (LGBM)
        print(f"\n1. Fitting base demand LGBM model on {len(df_train)} weekly observations...")
        self.base_demand_model.fit(df_train, df_val)
        
        # 2. Fit parametric elasticity model  
        print("\n2. Fitting double-log parametric model...")
        self.parametric_model.fit(df_train)
        
        self.fitted = True
        print("\nModel training completed!")
        
    def _store_historical_discount_patterns(self, df: pd.DataFrame):
        """Store historical discount patterns for each SKU for reference predictions"""
        for sku in df['sku'].unique():
            sku_data = df[df['sku'] == sku]
            discounts = sku_data['discount_pct'].values
            
            # Store various reference points
            self.sku_historical_discounts[sku] = {
                'mean_discount': float(np.mean(discounts)),
                'median_discount': float(np.median(discounts)),
                'mode_discount': float(discounts[np.argmax(np.bincount(np.round(discounts * 100).astype(int)))]) / 100,
                'most_common_discounts': list(np.unique(np.round(discounts, 2))),
                'min_discount': float(np.min(discounts)),
                'max_discount': float(np.max(discounts))
            }
    
    def _get_reference_discount(self, sku: str, target_discount: float) -> float:
        """Get the best historical reference discount for a SKU"""
        if sku not in self.sku_historical_discounts:
            return 0.0  # Fallback to 0% if no history
        
        historical = self.sku_historical_discounts[sku]
        available_discounts = historical['most_common_discounts']
        
        # Find the closest historical discount to target
        closest_discount = min(available_discounts, key=lambda x: abs(x - target_discount))
        
        # If target is very close to a historical discount, use it directly
        if abs(closest_discount - target_discount) < 0.02:  # Within 2%
            return closest_discount
        
        # Otherwise, use the most common discount (mode) as reference
        return historical['mode_discount']
        
    def predict(self, df: pd.DataFrame, discount_pct: Optional[np.ndarray] = None, 
                return_components: bool = False, use_reference_discount: bool = False):
        """
        Generate counterfactual predictions using cleaner approach:
        
        Args:
            df: Input dataframe
            discount_pct: Target discount percentages for counterfactual prediction
            return_components: Whether to return individual components
            use_reference_discount: If True, use 0% discount for LGBM baseline (better for counterfactuals)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        # Determine target discount
        if discount_pct is None:
            discount_pct = df['discount_pct'].values
        
        target_discount_ratios = 1 - discount_pct  # 20% discount -> 0.8 ratio
        
        if use_reference_discount:
            # For counterfactual prediction: LGBM predicts at best historical reference discount
            df_reference = df.copy()
            reference_discounts = []
            
            for idx, (i, row) in enumerate(df.iterrows()):
                sku = row['sku']
                if hasattr(discount_pct, '__len__') and len(discount_pct) > idx:
                    target_discount = discount_pct[idx]
                elif hasattr(discount_pct, '__len__'):
                    target_discount = discount_pct[0] if len(discount_pct) > 0 else row['discount_pct']
                else:
                    target_discount = discount_pct if discount_pct is not None else row['discount_pct']
                
                # Get best historical reference for this SKU
                ref_discount = self._get_reference_discount(sku, target_discount)
                reference_discounts.append(ref_discount)
            
            df_reference['discount_pct'] = reference_discounts
            historical_demand = self.base_demand_model.predict(df_reference)
            historical_discount_ratios = 1 - np.array(reference_discounts)
            
            print(f"Using historical reference discounts: mean={np.mean(reference_discounts):.3f}, "
                  f"range=[{np.min(reference_discounts):.3f}, {np.max(reference_discounts):.3f}]")
        else:
            # Standard prediction: LGBM uses the discount in the data
            historical_demand = self.base_demand_model.predict(df)
            historical_discount_ratios = 1 - df['discount_pct'].values
        
        # Apply paper's Equation (8): ln Y_i,t(d_i) = θ̂^T L̂_i (ln d_i - ln d_i^o) + ln Y_i,t^o
        ln_historical_demand = np.log(historical_demand + 1e-8)  # ln Y_i,t^o
        
        # Get parametric effects for counterfactual vs historical discounts
        ln_target_effect = self.parametric_model.predict_log_ratio(df, target_discount_ratios)    # g(d_i)
        ln_historical_effect = self.parametric_model.predict_log_ratio(df, historical_discount_ratios)  # g(d_i^o)
        
        # Paper's equation: difference in parametric effects
        ln_parametric_adjustment = ln_target_effect - ln_historical_effect  # θ̂^T L̂_i (ln d_i - ln d_i^o)
        
        # Final counterfactual prediction
        ln_counterfactual_demand = ln_historical_demand + ln_parametric_adjustment
        
        # Convert back to actual demand
        counterfactual_demand = np.exp(ln_counterfactual_demand)
        
        if return_components:
            return counterfactual_demand, historical_demand, ln_parametric_adjustment
        
        return counterfactual_demand
    
    def forecast(self, df: pd.DataFrame, horizon_weeks: int = 4, 
                 future_discounts: Optional[Dict[str, List[float]]] = None) -> pd.DataFrame:
        """Generate weekly forecasts with counterfactual discounts
        
        Args:
            df: Historical data (will be aggregated to weekly if not already)
            horizon_weeks: Number of weeks to forecast ahead
            future_discounts: Optional discount schedules by SKU
        """
        # Always aggregate to weekly level
        if 'week_of_year' not in df.columns:
            df = aggregate_to_weekly(df)
        
        last_date = pd.to_datetime(df['date']).max()
        last_data = df[df['date'] == last_date].copy()
        
        forecasts = []
        
        for h in range(1, horizon_weeks + 1):
            # Move forward by weeks, not days
            forecast_date = last_date + pd.Timedelta(weeks=h)
            forecast_data = last_data.copy()
            forecast_data['date'] = forecast_date
            
            # Update calendar features for weekly data
            forecast_data['week_of_year'] = forecast_date.isocalendar()[1]
            forecast_data['month'] = forecast_date.month
            forecast_data['quarter'] = forecast_date.quarter
            
            # Apply future discounts if provided
            if future_discounts:
                for sku, discounts in future_discounts.items():
                    if h-1 < len(discounts):
                        forecast_data.loc[forecast_data['sku'] == sku, 'discount_pct'] = discounts[h-1]
            
            # Get counterfactual predictions
            predictions = self.predict(forecast_data)
            
            forecast_data['forecast'] = predictions
            forecast_data['horizon_week'] = h
            forecasts.append(forecast_data)
        
        return pd.concat(forecasts, ignore_index=True)
    
    def get_elasticities(self) -> Dict:
        """Get estimated price elasticities"""
        return {
            'category': self.parametric_model.elasticities,
            'model_parameters': {
                'theta': self.parametric_model.theta.tolist() if self.parametric_model.theta is not None else None,
                'intercept': self.parametric_model.c
            }
        }
    
    def save(self, path: str):
        """Save model"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str):
        """Load model"""
        with open(path, 'rb') as f:
            return pickle.load(f)