#!/usr/bin/env python3
"""
Database initialization script for the Commodity Price Prediction app.
This script creates the database tables and can be used to populate sample data.
"""

from app import app, db, PredictionSearch
from datetime import datetime, timedelta
import random

def init_database():
    """Initialize the database and create all tables."""
    with app.app_context():
        # Create all tables
        db.create_all()
        print("✅ Database tables created successfully!")
        
        # Check if we already have some data
        existing_searches = PredictionSearch.query.count()
        if existing_searches > 0:
            print(f"ℹ️  Database already contains {existing_searches} search records.")
            return
        
        # Add some sample search data for demonstration
        sample_commodities = [
            "Tomato Big(Nepali)",
            "Potato Red", 
            "Onion Dry",
            "Cabbage",
            "Carrot(Local)",
            "Brinjal Long"
        ]
        
        print("🔄 Adding sample search data...")
        
        # Create sample searches over the last few days
        for i in range(15):
            search_date = datetime.utcnow() - timedelta(days=random.randint(0, 7))
            commodity = random.choice(sample_commodities)
            forecast_days = random.choice([3, 7, 14, 21, 30])
            # Generate price in NPR and convert to PHP
            avg_price_npr = round(random.uniform(20.0, 150.0), 2)
            avg_price = round(avg_price_npr * 0.40, 2)  # Convert to PHP
            
            sample_search = PredictionSearch(
                commodity_name=commodity,
                forecast_days=forecast_days,
                search_date=search_date,
                avg_predicted_price=avg_price
            )
            
            db.session.add(sample_search)
        
        db.session.commit()
        print("✅ Sample data added successfully!")
        print(f"📊 Total search records: {PredictionSearch.query.count()}")

if __name__ == '__main__':
    init_database() 