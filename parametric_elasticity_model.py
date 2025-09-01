"""
Parametric Price Elasticity Model

Implementation of the double-log parametric model from KDD 2021 paper:
"Markdowns in E-Commerce Fresh Retail: A Counterfactual Prediction and Multi-Period Optimization Approach"

This module provides category-level price elasticity estimation with hierarchical normalization
for improved stability and accuracy in counterfactual demand prediction.

Mathematical Foundation:
    Training: ln(Y_i / Y_norm) = (θ_1 + θ_2^T L_i) × ln(d_i) + c
    Inference: ln Y_i,t(d_i) = θ^T L̂_i (ln d_i - ln d_i,t^o) + ln Y_i,t^o

Where:
    Y_i: Demand for product i
    Y_norm: Normalization constant (category/product/global average)
    d_i: Discount ratio (1 - discount_percentage)
    L_i: Category one-hot encoding vector
    θ_1: Base elasticity parameter
    θ_2: Category-specific elasticity adjustments
    c: Intercept term

Key Features:
    - Hierarchical normalization (product/category/global levels)
    - Category-level elasticity estimation
    - Regularized least squares fitting
    - Counterfactual demand prediction

Author: Based on Alibaba Group's KDD 2021 research
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import pickle


class ParametricElasticityModel:
    """
    Double-log parametric model for price elasticity estimation.
    
    Implements the parametric component from the KDD 2021 paper with hierarchical
    normalization for improved stability. Estimates category-level price elasticities
    using regularized least squares on log-transformed normalized demand data.
    
    The model learns elasticity parameters that capture how demand responds to
    price changes, enabling counterfactual prediction at different discount levels.
    
    Attributes:
        regularization_strength (float): L2 regularization parameter for parameter fitting
        use_normalization (bool): Whether to normalize demand before log transformation
        normalization_method_ (str): Normalization approach ('category_mean', 'product_mean', 'global_mean')
        theta_1_ (float): Fitted base elasticity parameter
        theta_2_ (array): Fitted category-specific elasticity adjustments
        intercept_ (float): Fitted intercept term
        category_elasticities_ (dict): Final category-level elasticity values
        Y_norm_ (dict/float): Normalization constants used during training
    """
    
    def __init__(
        self,
        regularization_strength: float = 0.1,
        convergence_tolerance: float = 1e-6,
        max_iterations: int = 1000,
        use_normalization: bool = True
    ):
        """
        Initialize the parametric elasticity model.
        
        Args:
            regularization_strength: L2 regularization parameter (λ)
            convergence_tolerance: Convergence criterion for optimization
            max_iterations: Maximum iterations for fitting
            use_normalization: Whether to normalize demand before log transform
        """
        self.regularization_strength = regularization_strength
        self.convergence_tolerance = convergence_tolerance
        self.max_iterations = max_iterations
        self.use_normalization = use_normalization
        
        # Model parameters (to be learned)
        self.theta_1_ = None  # Base elasticity
        self.theta_2_ = None  # Category-specific adjustments
        self.intercept_ = None  # Intercept term
        
        # Category mappings
        self.category_encoder_ = None
        self.categories_ = None
        self.n_categories_ = 0
        
        # Normalization parameters
        self.Y_norm_ = None  # Normalization constant(s)
        self.normalization_method_ = 'category_mean'  # Options: 'product_mean', 'category_mean', 'global_mean'
        
        # Fitted elasticities
        self.category_elasticities_ = {}
    
    def _apply_normalization(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """
        Apply normalization to demand values based on the configured method.
        
        Normalizes demand values by dividing by appropriate normalization constants
        to improve numerical stability of log transformation and parameter estimation.
        
        Args:
            X (pd.DataFrame): Feature matrix containing 'category' and optionally 'sku' columns
            y (pd.Series): Demand values to normalize
            
        Returns:
            pd.Series: Normalized demand values (y / y_norm)
            
        Note:
            Normalization method is determined by self.normalization_method_:
            - 'product_mean': Uses product-specific averages (may be unstable)
            - 'category_mean': Uses category averages (recommended)
            - 'global_mean': Uses global average (most stable)
        """
        if isinstance(self.Y_norm_, dict):
            if self.normalization_method_ == 'product_mean' and 'sku' in X.columns:
                # Normalize by product-specific mean
                y_norm = X['sku'].map(self.Y_norm_).fillna(self.Y_norm_.get('default', y.mean()))
            elif self.normalization_method_ == 'category_mean':
                # Normalize by category-specific mean
                y_norm = X['category'].map(self.Y_norm_).fillna(y.mean())
            else:
                y_norm = y.mean()  # fallback
        else:
            # Single normalization constant
            y_norm = self.Y_norm_
        
        return y / y_norm
    
    def _get_normalization_value(self, category: str, sku: str = None) -> float:
        """
        Retrieve the appropriate normalization constant for a given product/category.
        
        Args:
            category (str): Product category
            sku (str, optional): Product SKU identifier
            
        Returns:
            float: Normalization constant for the specified product/category
            
        Note:
            Returns the normalization constant that was computed during training.
            Used internally for consistency, though normalization cancels out in inference.
        """
        if isinstance(self.Y_norm_, dict):
            if self.normalization_method_ == 'product_mean' and sku and sku in self.Y_norm_:
                return self.Y_norm_[sku]
            elif self.normalization_method_ == 'category_mean' and category in self.Y_norm_:
                return self.Y_norm_[category]
            else:
                # Fallback to first available value
                return next(iter(self.Y_norm_.values()))
        else:
            return self.Y_norm_
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ParametricElasticityModel':
        """
        Fit the parametric elasticity model with optional normalization.
        
        Paper equation with normalization:
        ln(Y_i / Y_norm) = (θ_1 + θ_2^T L_i) * ln(d_i) + c
        
        Args:
            X: Features DataFrame with columns ['category', 'discount_ratio']
               where discount_ratio = 1 - discount_pct
            y: Target variable (demand)
        
        Returns:
            Fitted model instance
        """
        # Validate inputs
        if 'category' not in X.columns or 'discount_ratio' not in X.columns:
            raise ValueError("X must contain 'category' and 'discount_ratio' columns")
        
        # Filter out zero or negative discount ratios (avoid log(0))
        valid_mask = (X['discount_ratio'] > 0) & (y > 0)
        X_valid = X[valid_mask].copy()
        y_valid = y[valid_mask].copy()
        
        # Calculate normalization constant(s)
        if self.use_normalization:
            if self.normalization_method_ == 'product_mean':
                # Y_norm_i = average sales of product i (may be unstable as paper mentions)
                self.Y_norm_ = {}
                if 'sku' in X_valid.columns:
                    for sku in X_valid['sku'].unique():
                        sku_data = y_valid[X_valid['sku'] == sku]
                        self.Y_norm_[sku] = sku_data.mean() if len(sku_data) > 0 else y_valid.mean()
                else:
                    self.Y_norm_ = y_valid.mean()
                
            elif self.normalization_method_ == 'category_mean':
                # Y_norm_category = average sales of category (more stable)
                self.Y_norm_ = {}
                for category in X_valid['category'].unique():
                    cat_data = y_valid[X_valid['category'] == category]
                    self.Y_norm_[category] = cat_data.mean() if len(cat_data) > 0 else y_valid.mean()
                    
            elif self.normalization_method_ == 'global_mean':
                # Y_norm = global average (most stable)
                self.Y_norm_ = y_valid.mean()
                
            else:
                # Fixed normalization value
                self.Y_norm_ = float(self.normalization_method_)
            
            # Apply normalization
            y_normalized = self._apply_normalization(X_valid, y_valid)
        else:
            self.Y_norm_ = 1.0
            y_normalized = y_valid
        
        # Encode categories
        self.categories_ = sorted(X_valid['category'].unique())
        self.n_categories_ = len(self.categories_)
        self.category_encoder_ = {cat: i for i, cat in enumerate(self.categories_)}
        
        # Create design matrix
        # L̂_i = [1, L_i^T] where L_i is one-hot encoding of category
        n_samples = len(X_valid)
        
        # Create augmented category matrix
        L_matrix = np.zeros((n_samples, self.n_categories_ + 1))
        L_matrix[:, 0] = 1  # First column is all ones for θ_1
        
        for idx, (_, row) in enumerate(X_valid.iterrows()):
            cat_idx = self.category_encoder_[row['category']]
            L_matrix[idx, cat_idx + 1] = 1
        
        # Log transform
        log_d = np.log(X_valid['discount_ratio'].values)
        log_y = np.log(y_normalized.values)  # Use normalized values
        
        # Create feature matrix: X_design[i,j] = L̂_i[j] * ln(d_i)
        X_design = L_matrix * log_d.reshape(-1, 1)
        
        # Solve regularized least squares
        # min ||log_y - X_design @ θ - c||² + λ||θ||²
        
        # Add intercept column
        X_with_intercept = np.column_stack([X_design, np.ones(n_samples)])
        
        # Regularization matrix (don't regularize intercept)
        reg_matrix = np.eye(self.n_categories_ + 2) * self.regularization_strength
        reg_matrix[-1, -1] = 0  # Don't regularize intercept
        
        # Solve: (X^T X + λI) θ = X^T y
        XtX = X_with_intercept.T @ X_with_intercept
        Xty = X_with_intercept.T @ log_y
        
        # Add regularization
        XtX_reg = XtX + reg_matrix
        
        # Solve for parameters
        try:
            params = np.linalg.solve(XtX_reg, Xty)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            params = np.linalg.pinv(XtX_reg) @ Xty
        
        # Extract parameters
        self.theta_1_ = params[0]  # Base elasticity
        self.theta_2_ = params[1:self.n_categories_ + 1]  # Category adjustments
        self.intercept_ = params[-1]  # Intercept
        
        # Calculate category-specific elasticities
        for cat, idx in self.category_encoder_.items():
            self.category_elasticities_[cat] = self.theta_1_ + self.theta_2_[idx]
        
        # Report fitting results
        self._report_fit_results(X_valid, y_normalized, log_y, X_with_intercept, params)
        
        return self
    
    def predict_multiplier(
        self,
        current_discount: float,
        target_discount: float,
        category: str
    ) -> float:
        """
        Calculate the demand multiplier for changing from current to target discount.
        
        Implements the core counterfactual prediction using the paper's formulation.
        The multiplier represents how much demand changes when moving from one
        discount level to another, based on learned price elasticity.
        
        Mathematical formula:
            Multiplier = exp(θ × (ln(d_target) - ln(d_current)))
            where d = 1 - discount_percentage
        
        Args:
            current_discount (float): Current discount percentage (0-1, e.g., 0.2 for 20%)
            target_discount (float): Target discount percentage (0-1, e.g., 0.3 for 30%)
            category (str): Product category for elasticity lookup
        
        Returns:
            float: Demand multiplier (>1 means demand increases, <1 means decrease)
            
        Example:
            >>> model.predict_multiplier(0.1, 0.2, 'Electronics')  # 10% → 20% discount
            1.15  # 15% demand increase
        """
        if self.theta_1_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get category elasticity
        if category not in self.category_elasticities_:
            print(f"Warning: Unknown category '{category}'. Using base elasticity.")
            elasticity = self.theta_1_
        else:
            elasticity = self.category_elasticities_[category]
        
        # Calculate discount ratios
        d_current = max(0.01, 1 - current_discount)  # Avoid log(0)
        d_target = max(0.01, 1 - target_discount)
        
        # Calculate multiplier
        log_ratio = np.log(d_target) - np.log(d_current)
        multiplier = np.exp(elasticity * log_ratio)
        
        return multiplier
    
    def adjust_demand(
        self,
        base_demand: float,
        current_discount: float,
        target_discount: float,
        category: str
    ) -> float:
        """
        Adjust demand from current discount level to target discount level.
        
        Applies the learned price elasticity to predict how demand changes
        when moving from one discount scenario to another. This is the primary
        method for counterfactual demand prediction.
        
        Args:
            base_demand (float): Current demand at current_discount level
            current_discount (float): Current discount percentage (0-1)
            target_discount (float): Target discount percentage (0-1)
            category (str): Product category for elasticity lookup
        
        Returns:
            float: Predicted demand at target discount level
            
        Example:
            >>> model.adjust_demand(1000, 0.0, 0.2, 'Electronics')
            1150.0  # 15% increase when adding 20% discount
        """
        multiplier = self.predict_multiplier(current_discount, target_discount, category)
        return base_demand * multiplier
    
    def get_elasticity(self, category: str) -> float:
        """
        Retrieve the fitted price elasticity parameter for a given category.
        
        Returns the category-specific elasticity (θ_1 + θ_2_category) that was
        learned during model training. Elasticity represents the percentage
        change in demand for a 1% change in discount ratio.
        
        Args:
            category (str): Product category name
        
        Returns:
            float: Price elasticity coefficient (typically negative)
                  - Values closer to 0: Less price-sensitive (inelastic)
                  - Values more negative: More price-sensitive (elastic)
                  
        Example:
            >>> model.get_elasticity('Electronics')
            -0.85  # Slightly elastic category
        """
        if category not in self.category_elasticities_:
            print(f"Warning: Unknown category '{category}'. Using base elasticity.")
            return self.theta_1_
        return self.category_elasticities_[category]
    
    def _report_fit_results(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        log_y: np.ndarray,
        X_design: np.ndarray,
        params: np.ndarray
    ):
        """Report fitting results and statistics."""
        # Calculate predictions
        log_y_pred = X_design @ params
        
        # Calculate R-squared
        ss_res = np.sum((log_y - log_y_pred) ** 2)
        ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        print("\nParametric Model Fitting Results:")
        print("=" * 50)
        print(f"Base elasticity (θ₁): {self.theta_1_:.4f}")
        print(f"Intercept (c): {self.intercept_:.4f}")
        print(f"R² (log space): {r2:.4f}")
        print(f"\nCategory-specific elasticities:")
        for cat in sorted(self.category_elasticities_.keys()):
            print(f"  {cat}: {self.category_elasticities_[cat]:.4f}")
    
    def save(self, filepath: str) -> None:
        """
        Save the fitted model to disk using pickle serialization.
        
        Persists the complete model state including fitted parameters,
        normalization constants, and category mappings for later use in production.
        
        Args:
            filepath (str): Path where the model should be saved (typically .pkl extension)
            
        Raises:
            ValueError: If model has not been fitted yet
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'ParametricElasticityModel':
        """
        Load a previously fitted model from disk.
        
        Deserializes a model that was saved using the save() method,
        restoring all fitted parameters and configuration.
        
        Args:
            filepath (str): Path to the saved model file
            
        Returns:
            ParametricElasticityModel: Loaded model instance ready for inference
            
        Raises:
            FileNotFoundError: If the model file doesn't exist
            pickle.PickleError: If the file is corrupted or incompatible
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model


def calculate_elasticity_from_data(
    df: pd.DataFrame,
    price_col: str = 'price',
    demand_col: str = 'demand',
    category_col: str = 'category'
) -> Dict[str, float]:
    """
    Calculate empirical price elasticities from historical data using log-log regression.
    
    This utility function provides a simple method to estimate price elasticities
    directly from price-demand data for validation or comparison purposes.
    Uses basic log-log regression: ln(demand) = elasticity × ln(price) + constant
    
    Args:
        df (pd.DataFrame): Historical sales data
        price_col (str): Column name containing price values
        demand_col (str): Column name containing demand/sales values  
        category_col (str): Column name containing product categories
    
    Returns:
        Dict[str, float]: Dictionary mapping category names to elasticity estimates
        
    Note:
        This is a simplified approach compared to the full parametric model.
        Used primarily for validation and sanity checking of main model results.
        Requires at least 10 observations per category for meaningful results.
    """
    elasticities = {}
    
    for category in df[category_col].unique():
        cat_data = df[df[category_col] == category].copy()
        
        # Filter valid data
        valid_mask = (cat_data[price_col] > 0) & (cat_data[demand_col] > 0)
        cat_data = cat_data[valid_mask]
        
        if len(cat_data) < 10:  # Need minimum samples
            continue
        
        # Log-log regression
        log_price = np.log(cat_data[price_col])
        log_demand = np.log(cat_data[demand_col])
        
        # Simple linear regression
        X = np.column_stack([log_price, np.ones(len(log_price))])
        try:
            params = np.linalg.lstsq(X, log_demand, rcond=None)[0]
            elasticity = params[0]  # Coefficient of log(price)
            elasticities[category] = elasticity
        except:
            continue
    
    return elasticities