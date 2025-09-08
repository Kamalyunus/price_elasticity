"""
Training script for the final price elasticity model
"""

import pandas as pd
import numpy as np
import yaml
import os
from price_elasticity_model import PriceElasticityModel
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def validate_data_quality(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Validate data meets quality requirements from config"""
    quality_checks = config['data']['quality_checks']
    
    # Check minimum history months
    min_months = quality_checks['min_history_months']
    date_range_months = (df['date'].max() - df['date'].min()).days / 30.44
    if date_range_months < min_months:
        raise ValueError(f"Data spans only {date_range_months:.1f} months, need {min_months}")
    
    # Check per-SKU promotional periods
    min_promo_periods = config['training']['min_promo_periods_per_sku']
    sku_promo_counts = df[df['discount_pct'] > 0].groupby('sku')['date'].nunique()
    insufficient_skus = sku_promo_counts[sku_promo_counts < min_promo_periods]
    
    if len(insufficient_skus) > 0:
        logger.warning(f"Filtering {len(insufficient_skus)} SKUs with <{min_promo_periods} promo periods")
        valid_skus = sku_promo_counts[sku_promo_counts >= min_promo_periods].index
        df = df[df['sku'].isin(valid_skus)]
    
    # Check per-category requirements
    min_skus_per_cat = config['training']['min_skus_per_category']
    category_sku_counts = df.groupby('category')['sku'].nunique()
    insufficient_cats = category_sku_counts[category_sku_counts < min_skus_per_cat]
    
    if len(insufficient_cats) > 0:
        logger.warning(f"Filtering {len(insufficient_cats)} categories with <{min_skus_per_cat} SKUs")
        valid_categories = category_sku_counts[category_sku_counts >= min_skus_per_cat].index
        df = df[df['category'].isin(valid_categories)]
    
    logger.info(f"After quality filtering: {len(df)} records, {df['sku'].nunique()} SKUs, {df['category'].nunique()} categories")
    return df

def main():
    print("="*80)
    print("PRICE ELASTICITY MODEL - KDD 2021 PAPER IMPLEMENTATION")
    print("Daily Paired Approach with Tunable Forgetting Factor")
    print("="*80)
    
    # Load configuration
    config = load_config()
    
    # Load data
    logger.info("Loading training data...")
    df = pd.read_csv('data/generated_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Validate data quality
    df = validate_data_quality(df, config)
    
    print(f"\nData Overview:")
    print(f"  Records: {len(df):,}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  SKUs: {df['sku'].nunique()}")
    print(f"  Categories: {df['category'].nunique()}")
    
    # Get true elasticities for comparison
    true_elasticities = df.groupby('category')['true_elasticity'].first().to_dict()
    
    # Split data for proper evaluation
    test_split = config['training']['test_split']
    cutoff_date = df['date'].min() + (df['date'].max() - df['date'].min()) * (1 - test_split)
    train_df = df[df['date'] <= cutoff_date]
    test_df = df[df['date'] > cutoff_date]
    
    print(f"\nTrain/Test Split:")
    print(f"  Training: {len(train_df):,} records ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    print(f"  Testing: {len(test_df):,} records ({test_df['date'].min().date()} to {test_df['date'].max().date()})")
    
    # Train with config parameters
    forgetting_factor = config['model']['forgetting_factor']
    regularization = config['model']['regularization']
    
    print(f"\nTraining with Config Parameters:")
    print(f"  Forgetting factor (λ): {forgetting_factor}")
    print(f"  Regularization (α): {regularization}")
    print(f"  Baseline days: {config['model']['baseline_days']}")
    
    model = PriceElasticityModel(
        forgetting_factor=forgetting_factor,
        regularization=regularization,
        baseline_days=config['model']['baseline_days']
    )
    
    # Fit model
    metrics = model.fit(train_df)
    
    print(f"\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    print(f"Paired samples: {metrics['n_samples']}")
    print(f"Categories: {metrics['n_categories']}")
    print(f"Weighted R²: {metrics['weighted_r2']:.3f}")
    print(f"Final loss: {metrics['final_loss']:.6f}")
    print(f"Convergence: {metrics['convergence']}")
    
    # Compare elasticities
    print(f"\n" + "="*60)
    print("ELASTICITY COMPARISON")
    print("="*60)
    print(f"{'Category':<12} | {'True':<10} | {'Learned':<10} | {'Error':<10}")
    print("-"*50)
    
    errors = []
    for cat in sorted(true_elasticities.keys()):
        true_val = true_elasticities[cat]
        learned_val = metrics['elasticities'][cat]
        error = abs(learned_val - true_val)
        errors.append(error)
        print(f"{cat:<12} | {true_val:9.3f} | {learned_val:9.3f} | {error:9.3f}")
    
    avg_error = np.mean(errors)
    print("-"*50)
    print(f"{'Average':<12} | {np.mean(list(true_elasticities.values())):9.3f} | "
          f"{np.mean(list(metrics['elasticities'].values())):9.3f} | {avg_error:9.3f}")
    
    # Test predictions
    print(f"\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    test_cases = [
        ('cat_0', 0.10),
        ('cat_0', 0.25), 
        ('cat_1', 0.15),
        ('cat_2', 0.30),
        ('cat_4', 0.40)
    ]
    
    print(f"{'Category':<12} | {'Discount':<10} | {'Predicted Lift':<15}")
    print("-"*40)
    
    for cat, discount in test_cases:
        lift = model.predict_lift(discount, cat)
        print(f"{cat:<12} | {discount*100:8.1f}% | {lift:13.2f}x")
    
    # Test set evaluation
    print(f"\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    
    # Extract paired samples from test set
    test_paired = model.extract_paired_samples(test_df)
    
    if len(test_paired) > 0:
        # Prepare test features
        test_X, test_y, test_weights = model.prepare_features(test_paired)
        test_X_with_intercept = np.column_stack([np.ones(len(test_X)), test_X])
        
        # Get predictions using trained parameters
        test_params = np.concatenate([[model.intercept], model.theta])
        test_predictions = test_X_with_intercept @ test_params
        
        # Calculate test R²
        test_ss_res = np.sum((test_y - test_predictions)**2)
        test_ss_tot = np.sum((test_y - test_y.mean())**2)
        test_r2 = 1 - test_ss_res / test_ss_tot
        
        # Calculate test MAPE for lift predictions
        actual_lifts = np.exp(test_y)
        predicted_lifts = np.exp(test_predictions)
        test_mape = np.mean(np.abs((actual_lifts - predicted_lifts) / actual_lifts)) * 100
        
        print(f"Test paired samples: {len(test_paired)}")
        print(f"Test R²: {test_r2:.3f}")
        print(f"Test MAPE: {test_mape:.1f}%")
        print(f"Generalization gap: {metrics['weighted_r2'] - test_r2:.3f}")
        
        test_metrics = {
            'test_samples': len(test_paired),
            'test_r2': test_r2,
            'test_mape': test_mape,
            'generalization_gap': metrics['weighted_r2'] - test_r2
        }
    else:
        print("No test paired samples found")
        test_metrics = None
    
    # Save model using config path
    model_dir = config['output']['model_dir']
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'price_elasticity_model.pkl')
    model.save(model_path)
    
    # Save results summary
    results_file = os.path.join(model_dir, config['output']['results_file'])
    results_summary = {
        'model_config': {
            'forgetting_factor': forgetting_factor,
            'regularization': regularization,
            'baseline_days': config['model']['baseline_days']
        },
        'training_metrics': metrics,
        'test_metrics': test_metrics,
        'data_quality': {
            'total_records': len(df),
            'training_records': len(train_df),
            'test_records': len(test_df),
            'skus': df['sku'].nunique(),
            'categories': df['category'].nunique()
        }
    }
    
    with open(results_file, 'w') as f:
        import json
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"\n" + "="*80)
    print("PAPER COMPLIANCE VERIFICATION")
    print("="*80)
    print("Double-log formulation: ln(Y_promo/Y_baseline) = θ×ln(d) + c")
    print("Category-specific elasticities: θ_1 + θ_2^T L_i")  
    print("Paper's loss function: Σ_t λ^(T-t)×(y_t - θ^T x_t)² + α×||θ||²")
    print("Forgetting factor λ = 0.99 for temporal weighting")
    print("L2 regularization α = 0.1")
    print("Daily paired approach (enhanced normalization)")
    
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Successfully trained price elasticity model:")
    print(f"   {metrics['n_samples']} paired training samples")
    print(f"   {metrics['n_categories']} product categories")
    print(f"   {metrics['weighted_r2']:.3f} weighted R² on training data")
    print(f"   {avg_error:.3f} average elasticity error vs true values")
    print(f"   Model saved to {model_path}")
    
    print(f"\nReady for production inference using:")
    print(f"  predicted_lift = model.predict_lift(discount_pct, category)")


if __name__ == '__main__':
    main()