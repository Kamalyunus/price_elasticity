"""
Test script for the price elasticity model with production scenarios.
"""

import pandas as pd
import numpy as np
from price_elasticity_model import PriceElasticityModel

def main():
    print("=" * 60)
    print("TESTING PRICE ELASTICITY MODEL")
    print("=" * 60)
    
    try:
        # Load the trained model
        model = PriceElasticityModel.load('models/price_elasticity_model.pkl')
        print("✅ Model loaded successfully")
    except FileNotFoundError:
        print("❌ Model not found. Please run train_model.py first.")
        return
    
    # View model's elasticities
    print("\nTrained Category Elasticities:")
    print("-" * 40)
    elasticities = model.get_elasticities()
    for cat, elasticity in sorted(elasticities.items()):
        print(f"  {cat}: {elasticity:.4f}")
    
    # Test 1: Single prediction for each category
    print("\nTest 1: Impact of 20% discount on different categories")
    print("-" * 50)
    discount = 0.2  # 20% discount
    
    for cat in sorted(elasticities.keys()):
        lift = model.predict_lift(discount, cat)
        print(f"  {cat}: {lift:.2f}x lift ({(lift-1)*100:+.1f}% demand increase)")
    
    # Test 2: Production scenario with base forecasts
    print(f"\nTest 2: Production Integration with Base Forecasts")
    print("-" * 50)
    
    scenarios = [
        {'category': 'cat_0', 'base_forecast': 1000, 'discount': 0.15},
        {'category': 'cat_1', 'base_forecast': 500, 'discount': 0.25},
        {'category': 'cat_4', 'base_forecast': 2000, 'discount': 0.30},
    ]
    
    for scenario in scenarios:
        lift = model.predict_lift(scenario['discount'], scenario['category'])
        final_forecast = scenario['base_forecast'] * lift
        print(f"  {scenario['category']}: {scenario['base_forecast']} → "
              f"{final_forecast:.0f} units ({scenario['discount']*100:.0f}% discount)")
    
    # Test 3: Batch processing simulation
    print(f"\nTest 3: Batch Processing Simulation")
    print("-" * 50)
    
    # Create sample batch data
    batch_data = pd.DataFrame({
        'sku': [f'SKU_{i:03d}' for i in range(5)],
        'category': ['cat_0', 'cat_1', 'cat_2', 'cat_3', 'cat_4'],
        'base_forecast': [800, 1200, 600, 1500, 900],
        'discount': [0.10, 0.20, 0.15, 0.25, 0.35]
    })
    
    # Apply model predictions
    batch_data['lift'] = batch_data.apply(
        lambda row: model.predict_lift(row['discount'], row['category']), 
        axis=1
    )
    batch_data['final_forecast'] = batch_data['base_forecast'] * batch_data['lift']
    batch_data['lift_pct'] = (batch_data['lift'] - 1) * 100
    
    print(batch_data[['sku', 'category', 'base_forecast', 'discount', 'final_forecast', 'lift_pct']].to_string(index=False, float_format='%.0f'))
    
    # Test 4: Different discount levels for one category
    print(f"\nTest 4: Discount Sensitivity Analysis (cat_0)")
    print("-" * 50)
    
    discounts = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    print(f"{'Discount':<10} | {'Lift':<8} | {'% Increase':<12}")
    print("-" * 35)
    
    for discount in discounts:
        lift = model.predict_lift(discount, 'cat_0')
        increase = (lift - 1) * 100
        print(f"{discount*100:8.0f}% | {lift:7.2f}x | {increase:10.1f}%")
    
    print(f"\n✅ All tests completed successfully!")
    print(f"\nModel ready for production use:")
    print(f"  model = PriceElasticityModel.load('models/price_elasticity_model.pkl')")
    print(f"  lift = model.predict_lift(discount_pct=0.25, category='YourCategory')")


if __name__ == '__main__':
    main()