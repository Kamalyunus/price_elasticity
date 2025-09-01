"""
Test script for the trained elasticity model with realistic scenarios.
"""

import pandas as pd
from elasticity_inference import ElasticityInference

def main():
    print("=" * 60)
    print("TESTING PARAMETRIC ELASTICITY MODEL")
    print("=" * 60)
    
    # Load the trained model
    inference = ElasticityInference('models/elasticity_model.pkl')
    
    # View model's elasticities
    print("\nTrained Category Elasticities:")
    print("-" * 40)
    for cat in ['cat_0', 'cat_1', 'cat_2', 'cat_3', 'cat_4']:
        elasticity = inference.model.get_elasticity(cat)
        print(f"  {cat}: {elasticity:.4f}")
    
    # Test 1: Single prediction for each category
    print("\nTest 1: Impact of 20% discount on 1000 unit base forecast")
    print("-" * 40)
    base_forecast = 1000
    discount = 0.2  # 20% discount
    
    for cat in ['cat_0', 'cat_1', 'cat_2', 'cat_3', 'cat_4']:
        final = inference.predict(
            base_forecast=base_forecast,
            discount_pct=discount,
            category=cat
        )
        lift = (final / base_forecast - 1) * 100
        print(f"  {cat}: {final:.0f} units (+{lift:.1f}% demand lift)")
    
    # Test 2: Different discount levels for most elastic category
    print("\nTest 2: Category 'cat_0' (most elastic) at different discounts")
    print("-" * 40)
    category = 'cat_0'  # Most elastic category
    
    for discount_pct in [0.0, 0.1, 0.2, 0.3, 0.5]:
        final = inference.predict(
            base_forecast=1000,
            discount_pct=discount_pct,
            category=category
        )
        print(f"  {discount_pct*100:>3.0f}% discount: {final:>5.0f} units")
    
    # Test 3: Batch processing with realistic data
    print("\nTest 3: Batch Processing Multiple Products")
    print("-" * 40)
    
    batch_data = pd.DataFrame({
        'product_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
        'category': ['cat_0', 'cat_1', 'cat_2', 'cat_3', 'cat_4'],
        'base_forecast': [500, 1000, 750, 1200, 300],
        'discount_pct': [0.15, 0.10, 0.25, 0.20, 0.30]
    })
    
    results = inference.predict_batch(batch_data)
    
    # Calculate revenue impact (assuming base prices)
    base_prices = [50000, 30000, 40000, 25000, 60000]
    results['base_price'] = base_prices
    results['revenue_no_discount'] = results['base_forecast'] * results['base_price']
    results['revenue_with_discount'] = (
        results['final_forecast'] * 
        results['base_price'] * 
        (1 - results['discount_pct'])
    )
    results['revenue_change'] = (
        (results['revenue_with_discount'] / results['revenue_no_discount'] - 1) * 100
    )
    
    print("\nProduct Performance:")
    for _, row in results.iterrows():
        print(f"  {row['product_id']} ({row['category']}):")
        print(f"    Base: {row['base_forecast']:.0f} units")
        print(f"    With {row['discount_pct']*100:.0f}% discount: {row['final_forecast']:.0f} units")
        print(f"    Revenue impact: {row['revenue_change']:+.1f}%")
    
    # Test 4: Counterfactual comparison
    print("\nTest 4: Counterfactual Analysis")
    print("-" * 40)
    print("Question: What if we change from 10% to 20% discount?")
    
    # Current scenario: 10% discount
    current_demand = inference.predict(
        base_forecast=1000,
        discount_pct=0.1,
        category='cat_2',
        base_discount_pct=0.0
    )
    
    # New scenario: 20% discount
    new_demand = inference.predict(
        base_forecast=1000,
        discount_pct=0.2,
        category='cat_2',
        base_discount_pct=0.0
    )
    
    print(f"  Current (10% off): {current_demand:.0f} units")
    print(f"  Proposed (20% off): {new_demand:.0f} units")
    print(f"  Demand increase: +{new_demand - current_demand:.0f} units")
    print(f"  Percentage lift: +{(new_demand/current_demand - 1)*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()