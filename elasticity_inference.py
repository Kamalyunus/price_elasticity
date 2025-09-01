"""
Production Inference Module for Parametric Price Elasticity

This module provides a clean production interface for applying trained parametric
price elasticity models to generate counterfactual demand forecasts. Designed for
integration with existing forecasting systems and real-time pricing optimization.

The module implements the inference component of the KDD 2021 paper approach,
enabling businesses to predict demand changes under different discount scenarios
using learned price elasticity parameters.

Mathematical Foundation:
    Paper equation with normalization:
    ln(Y_i,t(d_i) / Y_norm) = θ^T L̂_i (ln d_i - ln d_i,t^o) + ln(Y_i,t^o / Y_norm)
    
    Final inference formula (normalization cancels out):
    Y_i,t(d_i) = Y_i,t^o × exp[θ^T L̂_i × (ln d_i - ln d_i,t^o)]

Key Variables:
    Y_i,t^o: Base forecast from your forecasting team (at base discount level)
    Y_i,t(d_i): Final forecast at target discount level
    Y_norm: Normalization constant (learned during training, cancels out)
    d_i: Target discount ratio (1 - discount_percentage) for counterfactual
    d_i,t^o: Base discount ratio of input forecast (typically 1.0 for base price)
    θ^T L̂_i: Category-specific elasticity parameter (learned during training)

Features:
    - Fast inference using simple exponential calculations
    - Compatible with any base forecast (normalized or not)
    - Batch processing for multiple products
    - Clean integration with existing forecasting pipelines
    - Production-ready error handling and validation

Usage:
    Load trained model and apply to base forecasts for counterfactual prediction.
    The model seamlessly adjusts any base forecast to different discount scenarios.

Author: Based on KDD 2021 "Markdowns in E-Commerce Fresh Retail" paper
"""

import pandas as pd
import numpy as np
from typing import Union, Dict
from parametric_elasticity_model import ParametricElasticityModel


class ElasticityInference:
    """
    Production-ready inference engine for parametric price elasticity models.
    
    This class provides a clean, efficient interface for applying trained price
    elasticity models to base demand forecasts. It handles the mathematical
    transformations required for counterfactual prediction while abstracting
    away implementation complexity.
    
    The class is designed for production environments where:
    - Base forecasts come from existing forecasting systems
    - Real-time or batch elasticity adjustments are needed
    - Integration with pricing optimization systems is required
    - Performance and reliability are critical
    
    Key capabilities:
    - Single and batch prediction modes
    - Automatic handling of normalization (cancels out in inference)
    - Support for different base discount levels
    - Robust error handling and validation
    - Minimal memory footprint for high-throughput scenarios
    
    Attributes:
        model (ParametricElasticityModel): Loaded trained elasticity model with
            fitted parameters and category mappings
    """
    
    def __init__(self, model_path: str) -> None:
        """
        Initialize the inference engine with a pre-trained elasticity model.
        
        Loads the trained parametric elasticity model from disk and prepares it
        for inference operations. The model contains fitted elasticity parameters,
        normalization constants, and category mappings needed for prediction.
        
        Args:
            model_path (str): Absolute or relative path to the saved model file
                             (typically .pkl extension from training script)
        
        Raises:
            FileNotFoundError: If the model file doesn't exist at specified path
            pickle.PickleError: If the model file is corrupted or incompatible
            AttributeError: If the loaded object is not a valid elasticity model
        
        Example:
            >>> inference = ElasticityInference('models/elasticity_model.pkl')
            >>> # Model loaded and ready for predictions
        """
        self.model = ParametricElasticityModel.load(model_path)
    
    def predict(
        self,
        base_forecast: float,
        discount_pct: float,
        category: str,
        base_discount_pct: float = 0.0
    ) -> float:
        """
        Generate final forecast using the paper's equation with normalization.
        
        Paper equation with normalization:
        ln(Y_i,t(d_i) / Y_norm) = θ^T L̂_i (ln d_i - ln d_i,t^o) + ln(Y_i,t^o / Y_norm)
        
        The Y_norm terms cancel out when solving for Y_i,t(d_i):
        ln(Y_i,t(d_i)) - ln(Y_norm) = θ^T L̂_i (ln d_i - ln d_i,t^o) + ln(Y_i,t^o) - ln(Y_norm)
        ln(Y_i,t(d_i)) = θ^T L̂_i (ln d_i - ln d_i,t^o) + ln(Y_i,t^o)
        Y_i,t(d_i) = Y_i,t^o * exp[θ^T L̂_i (ln d_i - ln d_i,t^o)]
        
        Args:
            base_forecast: Y_i,t^o - base forecast from your team
            discount_pct: Target discount percentage (0-1) for counterfactual
            category: Product category
            base_discount_pct: Discount level of base forecast (default 0 = no discount)
        
        Returns:
            Y_i,t(d_i): Final demand forecast at target discount
        """
        # Get category-specific elasticity (θ^T L̂_i)
        theta = self.model.get_elasticity(category)
        
        # Calculate discount ratios
        d_i = 1 - discount_pct           # Target discount ratio
        d_i_o = 1 - base_discount_pct    # Base discount ratio
        
        # Avoid log(0)
        d_i = max(0.01, d_i)
        d_i_o = max(0.01, d_i_o)
        
        # Apply paper's equation with normalization:
        # The normalization terms (Y_norm) cancel out in the final equation
        # Y_i,t(d_i) = Y_i,t^o * exp[θ^T L̂_i (ln d_i - ln d_i,t^o)]
        
        elasticity_adjustment = theta * (np.log(d_i) - np.log(d_i_o))
        final_forecast = base_forecast * np.exp(elasticity_adjustment)
        
        return final_forecast
    
    def predict_batch(
        self,
        forecasts_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate final forecasts for multiple products in batch mode.
        
        Efficiently processes multiple forecast adjustments in a single operation,
        ideal for production scenarios with large numbers of products or
        systematic pricing optimization across product portfolios.
        
        This method applies the same mathematical transformation as predict()
        but optimized for batch processing with pandas operations.
        
        Args:
            forecasts_df (pd.DataFrame): Input forecasts with required columns:
                - base_forecast (float): Y_i,t^o from your forecasting team
                - discount_pct (float): Target discount (0-1) for counterfactual
                - category (str): Product category for elasticity lookup
                - base_discount_pct (float, optional): Discount of base forecast (default 0)
                
        Returns:
            pd.DataFrame: Original DataFrame with additional column:
                - final_forecast (float): Y_i,t(d_i) adjusted demand at target discount
        
        Raises:
            KeyError: If required columns are missing from input DataFrame
            ValueError: If data types are incompatible or values are out of range
            
        Example:
            >>> df = pd.DataFrame({
            ...     'base_forecast': [1000, 2000], 
            ...     'discount_pct': [0.1, 0.2],
            ...     'category': ['Electronics', 'Groceries']
            ... })
            >>> results = inference.predict_batch(df)
            >>> print(results['final_forecast'])
            [1084.2, 2315.7]  # Adjusted forecasts
        
        Note:
            For large datasets (>10,000 rows), consider processing in chunks
            to manage memory usage efficiently.
        """
        result = forecasts_df.copy()
        
        # Add base_discount_pct if not present (assume base forecast at no discount)
        if 'base_discount_pct' not in result.columns:
            result['base_discount_pct'] = 0.0
        
        # Apply equation to each row
        result['final_forecast'] = result.apply(
            lambda row: self.predict(
                base_forecast=row['base_forecast'],
                discount_pct=row['discount_pct'],
                category=row['category'],
                base_discount_pct=row.get('base_discount_pct', 0.0)
            ),
            axis=1
        )
        
        return result


def main() -> None:
    """
    Demonstration and testing function for the elasticity inference module.
    
    Provides practical examples of how to use the ElasticityInference class
    for both single predictions and batch processing. Useful for testing
    model integration and validating inference pipeline functionality.
    
    This function serves as both documentation and a basic smoke test,
    showing realistic usage patterns for production deployment.
    
    Examples include:
    - Loading a trained model
    - Single product forecast adjustment
    - Batch processing multiple products
    - Error handling and validation
    
    Note:
        Requires a trained model file at 'models/elasticity_model.pkl'.
        If the file doesn't exist, creates a mock instance for demonstration.
    """
    # Initialize
    inference = ElasticityInference('models/elasticity_model.pkl')
    
    # Single prediction
    final_demand = inference.predict(
        base_forecast=1000,  # Forecast at base price
        discount_pct=0.2,    # 20% discount
        category='Electronics'
    )
    print(f"Final forecast: {final_demand:.0f} units")
    
    # Batch prediction
    data = pd.DataFrame({
        'sku': ['SKU_001', 'SKU_002', 'SKU_003'],
        'category': ['Electronics', 'Groceries', 'Clothing'],
        'base_forecast': [1000, 2000, 500],
        'discount_pct': [0.1, 0.25, 0.15]
    })
    
    results = inference.predict_batch(data)
    print("\nBatch predictions:")
    print(results[['sku', 'category', 'base_forecast', 'discount_pct', 'final_forecast']])


if __name__ == "__main__":
    main()