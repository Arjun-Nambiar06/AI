# app.py
# Main application using Streamlit for UI

import csv
import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator
from collections import defaultdict
import plotly as px
import matplotlib as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

 
row_count = 0

# Set page config
st.set_page_config(
    page_title="Real-Time Shop Demand Dashboard",
    page_icon="🛒",
    layout="wide",
)

#main()
# Data schema definitions
class ProductEvent(BaseModel):
    """Base schema for all product-related events"""
    event_id: str = Field(..., description="Unique identifier for this event")
    vendor_id: str = Field(..., description="Identifier for the vendor")
    timestamp: datetime = Field(default_factory=datetime.now, description="When this event occurred")
    location: Dict[str, float] = Field(..., description="Geographic coordinates {lat, lng}")
    product_id: str = Field(..., description="Unique product identifier")
    product_name: str = Field(..., description="Human-readable product name")
    category: str = Field(..., description="Product category")

class SaleEvent(ProductEvent):
    """Schema for product sales events"""
    event_type: str = Field(default="sale", Literal=True)
    quantity: int = Field(..., description="Number of units sold", gt=0)
    price_per_unit: float = Field(..., description="Price per unit")

class RestockEvent(ProductEvent):
    """Schema for product restock events"""
    event_type: str = Field(default="restock", Literal=True)
    quantity: int = Field(..., description="Number of units added to inventory", gt=0)

# Mock implementation of Fluvio client for MVP
class FluvioClient:
    def __init__(self, topic: str = "shop-events"):
        self.topic = topic
        self.callbacks = []
        print(f"Initialized Fluvio client for topic: {topic}")
        
    async def produce(self, event: Dict[str, Any]) -> None:
        """Send event to Fluvio topic"""
        print(f"Producing event to {self.topic}: {json.dumps(event)}")
        # In real implementation, this would send to Fluvio
        # For MVP, we'll simulate by directly triggering consumers
        for callback in self.callbacks:
            await callback(event)
        
    def consume(self, callback) -> None:
        """Register consumer callback"""
        self.callbacks.append(callback)
        print(f"Registered consumer for {self.topic}")

# Data processor for real-time analytics
class DemandProcessor:
    def __init__(self):
        # In-memory storage for MVP (would use a proper database in production)
        self.recent_sales = []
        self.trending_products = set()
        self.sales_by_product = defaultdict(int)
        self.sales_by_category = defaultdict(int)
        self.sales_by_location = defaultdict(int)
        self.trending_threshold = 5  # Flag as trending if > 5 sales in 15 min
        self.location_data = []  # For map visualization
        
    async def process_event(self, event: Dict[str, Any]) -> None:
        """Process incoming event"""
        if event.get("event_type") == "sale":
            await self._process_sale(event)
        elif event.get("event_type") == "restock":
            await self._process_restock(event)
            
    async def _process_sale(self, event: Dict[str, Any]) -> None:
        """Process a sale event"""
        # Add to recent sales
        self.recent_sales.append(event)
        
        # Update aggregations
        product_id = event["product_id"]
        category = event["category"]
        location_key = f"{event['location']['lat']:.2f},{event['location']['lng']:.2f}"
        quantity = event["quantity"]
        
        self.sales_by_product[product_id] += quantity
        self.sales_by_category[category] += quantity
        self.sales_by_location[location_key] += quantity
        
        # Update location data for map
        self.location_data.append({
            "lat": event["location"]["lat"],
            "lon": event["location"]["lng"],
            "size": quantity,
            "product": event["product_name"]
        })
        
        # Clean up old events (keep only last 15 minutes)
        current_time = datetime.now()
        cutoff = current_time - timedelta(minutes=15)
        self.recent_sales = [
            e for e in self.recent_sales 
            if datetime.fromisoformat(e["timestamp"]) > cutoff
        ]
        
        # Check for trending products
        self._update_trending_products()
    
    async def _process_restock(self, event: Dict[str, Any]) -> None:
        """Process a restock event"""
        # For MVP, we're just logging restock events
        print(f"Restock event received: {event['product_name']} - {event['quantity']} units")
    
    def _update_trending_products(self) -> None:
        """Update the set of trending products"""
        # Group sales by product in the last 15 minutes
        product_counts = defaultdict(int)
        for event in self.recent_sales:
            product_id = event["product_id"]
            product_counts[product_id] += event["quantity"]
        
        # Identify trending products
        self.trending_products = {
            product_id for product_id, count in product_counts.items()
            if count > self.trending_threshold
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get all data needed for the dashboard"""
        return {
            "trending_products": list(self.trending_products),
            "sales_by_product": dict(self.sales_by_product),
            "sales_by_category": dict(self.sales_by_category),
            "sales_by_location": dict(self.sales_by_location),
            "recent_sales": self.recent_sales[-50:],  # Last 50 sales
            "location_data": self.location_data      # For map visualization
        }

# Initialize our processor and Fluvio client
processor = DemandProcessor()
fluvio_client = FluvioClient()

# Set up async event handling
async def handle_event(event):
    await processor.process_event(event)

# Register event handler with Fluvio
fluvio_client.consume(handle_event)

# Function to submit events (will be called from the UI)
async def submit_event(event_data):
    event_id = f"evt_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    event_data["event_id"] = event_id
    await fluvio_client.produce(event_data)
    return {"status": "success", "event_id": event_id}

# Helper function to run async functions from Streamlit
def run_async(func, *args, **kwargs):
    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(func(*args, **kwargs))
    loop.close()
    return result

# Add sample data for demo purposes
def add_sample_data():
    products = [
        {"id": "prod_001", "name": "Organic Bananas", "category": "Produce", "price": 1.99},
        {"id": "prod_002", "name": "Whole Milk", "category": "Dairy", "price": 3.49},
        {"id": "prod_003", "name": "Sourdough Bread", "category": "Bakery", "price": 4.99},
        {"id": "prod_004", "name": "Ground Beef", "category": "Meat", "price": 6.99},
        {"id": "prod_005", "name": "Wireless Earbuds", "category": "Electronics", "price": 89.99},
        {"id": "prod_006", "name": "T-Shirt", "category": "Clothing", "price": 15.99},
    ]
    
    # Generate random coordinates around San Francisco
    base_lat, base_lng = 27.6693, 77.32292
    with open("shop_data.csv",'a',newline='') as f:
        writer = csv.writer(f)
        for _ in range(10):
            product = np.random.choice(products)
            lat = base_lat + (np.random.random() - 0.5) * 0.1
            lng = base_lng + (np.random.random() - 0.5) * 0.1
            quantity = np.random.randint(1, 10)
            
            event = {
                "event_type": "sale",
                "vendor_id": f"vendor_{np.random.randint(1, 6)}",
                "timestamp": datetime.now().isoformat(),
                "location": {"lat": lat, "lng": lng},
                "product_id": product["id"],
                "product_name": product["name"],
                "category": product["category"],
                "quantity": quantity,
                "price_per_unit": product["price"]
            }

            quadrant = [event["location"]["lat"],event["location"]["lng"]]
            if quadrant[0] - base_lat >= 0:
                if quadrant[1] - base_lng >= 0:
                    location = "Northeast"
                else:
                    location = "Northwest"
            else:
                if quadrant[1] - base_lng >= 0:
                    location = "Southeast"
                else:
                    location = "Southwest"

            writer.writerow([event["vendor_id"],event["timestamp"], event["location"], location, event["category"], event["product_id"], event["product_name"], event["price_per_unit"], event["quantity"], (event["price_per_unit"]*event["quantity"])])

    with open("shop_data.csv",'r') as f:
        rdr = csv.reader(f)
        for i in range(0, row_count+1):
            next(rdr)
        for i in rdr:
            event = {
            "event_type": "sale",
            "vendor_id": i[0],
            "timestamp": i[1],
            "location": eval(i[2]),
            "product_id": i[5],
            "product_name": i[6],
            "category": i[4],
            "quantity": eval(i[8]),
            "price_per_unit": eval(i[7])
            }
            row_count += 1
            run_async(submit_event, event)
            time.sleep(0.5)  # Brief delay between events

def predict_data():
    model = LinearRegression()
    df = pd.read_csv("shop_data.csv")
    enc1 = preprocessing.OneHotEncoder()
    enc1.fit(df[['ProductID']])
    one_hot = enc1.transform(df[['ProductID']]).toarray()
    df[['P1','P2','P3','P4','P5','P6']] = one_hot

    enc2 = preprocessing.OneHotEncoder()
    enc2.fit(df[['Location']])
    one_hot = enc2.transform(df[['Location']]).toarray()
    df[['Northeast','Northwest','Southeast','Southwest']] = one_hot

    enc3 = preprocessing.OneHotEncoder()
    enc3.fit(df[['Vendor']])
    one_hot = enc3.transform(df[['Vendor']]).toarray()
    df[['V1','V2','V3','V4','V5','V6']] = one_hot

    inputs = df[['P1','P2','P3','P4','P5','P6','Northeast','Northwest','Southeast','Southwest','V1','V2','V3','V4','V5','V6','ItemPrice','Quantity']]
    targets = df.Amount

    model.fit(inputs,targets)
    coeff = model.coef_
    print(coeff) #test

# Button to load sample data
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

if not st.session_state.initialized:
    if st.sidebar.button("Load Sample Data"):
        with st.spinner("Loading sample data..."):
            add_sample_data()
            st.session_state.initialized = True

# Streamlit UI
st.title("Real-Time Local Shop Demand Dashboard")

# Live indicator
col1, col2 = st.columns([1, 10])
with col1:
    st.markdown("### 🟢")
with col2:
    st.markdown("### Live Data")

# Set up tabs for different dashboard views
tab1, tab2, tab3 = st.tabs(["Dashboard", "Vendor Input", "Recent Events"])

with tab1:
    # Dashboard layout
    st.subheader("Sales Overview")
    
    # Top row with metrics
    metric_cols = st.columns(4)
    
    total_sales = sum(processor.sales_by_product.values())
    num_categories = len(processor.sales_by_category)
    num_products = len(processor.sales_by_product)
    num_trending = len(processor.trending_products)
    
    with metric_cols[0]:
        st.metric("Total Sales", total_sales)
    with metric_cols[1]:
        st.metric("Categories", num_categories)
    with metric_cols[2]:
        st.metric("Products", num_products)
    with metric_cols[3]:
        st.metric("Trending Products", num_trending)
    
    # Charts row
    chart_cols = st.columns(2)
    
    with chart_cols[0]:
        st.subheader("Sales by Category")
        if processor.sales_by_category:
            category_df = pd.DataFrame({
                "Category": list(processor.sales_by_category.keys()),
                "Sales": list(processor.sales_by_category.values())
            })
            st.bar_chart(category_df.set_index("Category"))
        else:
            st.info("No category data available yet")
    
    with chart_cols[1]:
        st.subheader("Top Products")
        if processor.sales_by_product:
            # Get top 5 products
            top_products = sorted(
                processor.sales_by_product.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            # Find product names
            product_names = {}
            for sale in processor.recent_sales:
                product_names[sale["product_id"]] = sale["product_name"]
            
            product_df = pd.DataFrame({
                "Product": [product_names.get(p[0], p[0]) for p in top_products],
                "Sales": [p[1] for p in top_products]
            })
            st.bar_chart(product_df.set_index("Product"))
        else:
            st.info("No product data available yet")
    
    # Map section
    st.subheader("Live Demand Heatmap")
    
    if processor.location_data:
        map_data = pd.DataFrame(processor.location_data)
        st.map(map_data)
    else:
        st.info("No location data available yet")
    
    # Hot products section
    st.subheader("Hot Products")
    
    if processor.trending_products:
        trending_cols = st.columns(len(processor.trending_products))
        
        for i, product_id in enumerate(processor.trending_products):
            # Find product details
            product_info = next(
                (sale for sale in processor.recent_sales if sale["product_id"] == product_id),
                None
            )
            
            if product_info:
                with trending_cols[i]:
                    st.markdown(f"""
                    ### 🔥 {product_info["product_name"]}
                    **Category:** {product_info["category"]}
                    
                    **Units Sold:** {processor.sales_by_product[product_id]}
                    """)
    else:
        st.info("No trending products detected yet")

with tab2:
    # Vendor input form
    st.header("Add New Event")
    
    with st.form("event_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            event_type = st.selectbox("Event Type", ["sale", "restock"])
            vendor_id = st.text_input("Vendor ID", value=f"vendor_{np.random.randint(1, 100)}")
            product_id = st.text_input("Product ID", value=f"prod_{np.random.randint(100, 999)}")
            product_name = st.text_input("Product Name")
        
        with col2:
            category = st.selectbox("Category", ["Produce", "Dairy", "Bakery", "Meat", "Electronics", "Clothing"])
            quantity = st.number_input("Quantity", min_value=1, value=1)
            price = st.number_input("Price Per Unit", min_value=0.01, value=9.99, step=0.01)
            
        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Latitude", value=37.7749, format="%.6f")
        with col2:
            lng = st.number_input("Longitude", value=-122.4194, format="%.6f")
        
        submit_button = st.form_submit_button("Submit Event")
        
        if submit_button:
            event_data = {
                "event_type": event_type,
                "vendor_id": vendor_id,
                "timestamp": datetime.now().isoformat(),
                "location": {"lat": lat, "lng": lng},
                "product_id": product_id,
                "product_name": product_name,
                "category": category,
                "quantity": quantity
            }
            
            if event_type == "sale":
                event_data["price_per_unit"] = price
            
            result = run_async(submit_event, event_data)
            
            if result["status"] == "success":
                st.success(f"Event submitted successfully! Event ID: {result['event_id']}")
            else:
                st.error("Failed to submit event")

with tab3:
    # Recent events table
    st.header("Recent Events")
    
    if processor.recent_sales:
        # Convert recent sales to DataFrame for display
        events_df = pd.DataFrame([
            {
                "Time": datetime.fromisoformat(e["timestamp"]).strftime("%H:%M:%S"),
                "Event": e["event_type"].capitalize(),
                "Product": e["product_name"],
                "Category": e["category"],
                "Quantity": e["quantity"],
                "Vendor": e["vendor_id"]
            }
            for e in processor.recent_sales
        ])
        
        st.dataframe(events_df, use_container_width=True)
    else:
        st.info("No events recorded yet")

# Auto-refresh the dashboard every 5 seconds
if st.sidebar.checkbox("Auto-refresh (10 min)", value=True):
    time.sleep(600)
    st.rerun()

# Sidebar controls
st.sidebar.header("Dashboard Controls")

# Trending threshold control
new_threshold = st.sidebar.slider(
    "Trending Threshold (sales in 15 min)",
    min_value=1,
    max_value=20,
    value=processor.trending_threshold
)

if new_threshold != processor.trending_threshold:
    processor.trending_threshold = new_threshold
    processor._update_trending_products()

# Add demo events
if st.sidebar.button("Add Random Sale Event"):
    categories = ["Produce", "Dairy", "Bakery", "Meat", "Electronics", "Clothing"]
    products = [
        {"id": "prod_001", "name": "Organic Bananas", "category": "Produce", "price": 1.99},
        {"id": "prod_002", "name": "Whole Milk", "category": "Dairy", "price": 3.49},
        {"id": "prod_003", "name": "Sourdough Bread", "category": "Bakery", "price": 4.99},
        {"id": "prod_004", "name": "Ground Beef", "category": "Meat", "price": 6.99},
        {"id": "prod_005", "name": "Wireless Earbuds", "category": "Electronics", "price": 89.99},
        {"id": "prod_006", "name": "T-Shirt", "category": "Clothing", "price": 15.99},
    ]
    
    product = np.random.choice(products)
    base_lat, base_lng = 37.7749, -122.4194
    lat = base_lat + (np.random.random() - 0.5) * 0.1
    lng = base_lng + (np.random.random() - 0.5) * 0.1
    
    event = {
        "event_type": "sale",
        "vendor_id": f"vendor_{np.random.randint(1, 6)}",
        "timestamp": datetime.now().isoformat(),
        "location": {"lat": lat, "lng": lng},
        "product_id": product["id"],
        "product_name": product["name"],
        "category": product["category"],
        "quantity": np.random.randint(1, 10),
        "price_per_unit": product["price"]
    }
    
    run_async(submit_event, event)
    st.sidebar.success("Random sale event added!")

# Information panel
with st.sidebar.expander("About this Dashboard"):
    st.markdown("""
    ### Real-Time Local Shop Demand Dashboard
    
    This dashboard provides real-time insights into local shop demand patterns.
    
    **Features:**
    - Live tracking of product sales
    - Geographic visualization of demand
    - Hot product detection
    - Category-based analytics
    
    **Built with:**
    - Streamlit
    - Fluvio for event streaming
    - Python for data processing
    """)
