"""
Smart Inventory Intelligence — Streamlit App
Run with: streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------
st.set_page_config(page_title="Smart Inventory Intelligence", layout="wide")
sns.set_style("whitegrid")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")


# ---------------------------------------------------------------
# Cached Loaders
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    demand = pd.read_csv(os.path.join(DATA_DIR, "demand_forecasting.csv"))
    inventory = pd.read_csv(os.path.join(DATA_DIR, "inventory_monitoring.csv"))
    pricing = pd.read_csv(os.path.join(DATA_DIR, "pricing_optimization.csv"))
    demand["Date"] = pd.to_datetime(demand["Date"])
    return demand, inventory, pricing


@st.cache_resource
def load_models():
    demand_model = joblib.load(os.path.join(MODEL_DIR, "demand_forecast_model.pkl"))
    stockout_model = joblib.load(os.path.join(MODEL_DIR, "stockout_risk_model.pkl"))
    pricing_model = joblib.load(os.path.join(MODEL_DIR, "pricing_model.pkl"))
    return demand_model, stockout_model, pricing_model


demand, inventory, pricing = load_data()

try:
    demand_model, stockout_model, pricing_model = load_models()
    models_loaded = True
except FileNotFoundError:
    models_loaded = False


# ---------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------
st.sidebar.title("📦 Smart Inventory Intelligence")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Demand Forecasting", "Inventory Monitoring", "Pricing Optimization"]
)

if not models_loaded:
    st.sidebar.warning("Model files not found in /models. Run the notebook first to train and save models.")


# ---------------------------------------------------------------
# Overview Page
# ---------------------------------------------------------------
if page == "Overview":
    st.title("Smart Inventory Intelligence Dashboard")
    st.markdown(
        "An end-to-end analytics system covering **demand forecasting**, "
        "**inventory risk monitoring**, and **pricing optimization**."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Products (Demand)", demand["Product ID"].nunique())
    col2.metric("Stores Tracked", demand["Store ID"].nunique())
    col3.metric("Records Analyzed", f"{len(demand):,}")

    st.subheader("Monthly Sales Trend")
    monthly = demand.groupby(demand["Date"].dt.to_period("M"))["Sales Quantity"].sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    monthly.plot(ax=ax, marker="o", color="steelblue")
    ax.set_ylabel("Total Sales Quantity")
    ax.set_xlabel("Month")
    st.pyplot(fig)

    st.subheader("Dataset Preview")
    tab1, tab2, tab3 = st.tabs(["Demand", "Inventory", "Pricing"])
    with tab1:
        st.dataframe(demand.head(20))
    with tab2:
        st.dataframe(inventory.head(20))
    with tab3:
        st.dataframe(pricing.head(20))


# ---------------------------------------------------------------
# Demand Forecasting Page
# ---------------------------------------------------------------
elif page == "Demand Forecasting":
    st.title("📈 Demand Forecasting")

    col1, col2 = st.columns(2)
    with col1:
        price = st.slider("Price ($)", float(demand["Price"].min()), float(demand["Price"].max()), 25.0)
        promo = st.selectbox("Promotion Active?", ["No", "Yes"])
        month = st.selectbox("Month", list(range(1, 13)))
    with col2:
        seasonality = st.selectbox("Seasonality Factor", demand["Seasonality Factors"].unique())
        external = st.selectbox("External Factor", demand["External Factors"].unique())
        segment = st.selectbox("Customer Segment", demand["Customer Segments"].unique())

    if st.button("Predict Demand"):
        if not models_loaded:
            st.error("Model not found — train and save it via the notebook first.")
        else:
            # Encode inputs the same way as training (simple positional mapping)
            promo_map = {v: i for i, v in enumerate(sorted(demand["Promotions"].astype(str).unique()))}
            season_map = {v: i for i, v in enumerate(sorted(demand["Seasonality Factors"].astype(str).unique()))}
            ext_map = {v: i for i, v in enumerate(sorted(demand["External Factors"].astype(str).unique()))}
            seg_map = {v: i for i, v in enumerate(sorted(demand["Customer Segments"].astype(str).unique()))}
            quarter = (month - 1) // 3 + 1
            day_of_week = 2  # neutral default (mid-week)

            input_df = pd.DataFrame([{
                "Price": price,
                "Promotions": promo_map.get(promo, 0),
                "Seasonality Factors": season_map.get(seasonality, 0),
                "External Factors": ext_map.get(external, 0),
                "Customer Segments": seg_map.get(segment, 0),
                "Month": month,
                "Quarter": quarter,
                "DayOfWeek": day_of_week
            }])

            pred = demand_model.predict(input_df)[0]
            st.success(f"Predicted Sales Quantity: **{pred:.0f} units**")

    st.subheader("Historical Sales by Seasonality")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(x="Seasonality Factors", y="Sales Quantity", data=demand, ax=ax)
    plt.xticks(rotation=30)
    st.pyplot(fig)


# ---------------------------------------------------------------
# Inventory Monitoring Page
# ---------------------------------------------------------------
elif page == "Inventory Monitoring":
    st.title("📊 Inventory Monitoring")

    inv = inventory.copy()
    inv["Stockout_Risk"] = np.where(inv["Stock Levels"] <= inv["Reorder Point"], "At Risk", "OK")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total SKUs", len(inv))
    col2.metric("At Risk of Stockout", int((inv["Stockout_Risk"] == "At Risk").sum()))
    col3.metric("Avg Lead Time (days)", f"{inv['Supplier Lead Time (days)'].mean():.1f}")

    st.subheader("Filter by Risk Status")
    risk_filter = st.selectbox("Show", ["All", "At Risk", "OK"])
    display_df = inv if risk_filter == "All" else inv[inv["Stockout_Risk"] == risk_filter]
    st.dataframe(display_df.sort_values("Stock Levels").head(50))

    st.subheader("Stock Levels vs Reorder Point")
    fig, ax = plt.subplots(figsize=(10, 4))
    sample = inv.sample(min(200, len(inv)), random_state=1)
    ax.scatter(sample["Reorder Point"], sample["Stock Levels"],
               c=np.where(sample["Stockout_Risk"] == "At Risk", "red", "green"), alpha=0.6)
    ax.plot([0, inv["Reorder Point"].max()], [0, inv["Reorder Point"].max()], "k--", linewidth=1)
    ax.set_xlabel("Reorder Point")
    ax.set_ylabel("Stock Levels")
    st.pyplot(fig)

    if models_loaded:
        st.subheader("Predict Stockout Risk for a New Entry")
        c1, c2, c3 = st.columns(3)
        with c1:
            stock = st.number_input("Stock Levels", min_value=0, value=200)
            lead_time = st.number_input("Supplier Lead Time (days)", min_value=0, value=10)
        with c2:
            freq = st.number_input("Stockout Frequency", min_value=0, value=5)
            reorder = st.number_input("Reorder Point", min_value=0, value=150)
        with c3:
            capacity = st.number_input("Warehouse Capacity", min_value=0, value=1000)
            fulfillment = st.number_input("Order Fulfillment Time (days)", min_value=0, value=5)

        if st.button("Predict Risk"):
            input_df = pd.DataFrame([{
                "Stock Levels": stock,
                "Supplier Lead Time (days)": lead_time,
                "Stockout Frequency": freq,
                "Reorder Point": reorder,
                "Warehouse Capacity": capacity,
                "Order Fulfillment Time (days)": fulfillment
            }])
            risk_pred = stockout_model.predict(input_df)[0]
            label = "⚠️ At Risk of Stockout" if risk_pred == 1 else "✅ Stock Level OK"
            st.success(label)


# ---------------------------------------------------------------
# Pricing Optimization Page
# ---------------------------------------------------------------
elif page == "Pricing Optimization":
    st.title("💰 Pricing Optimization")

    pr = pricing.copy()
    pr["Price_Gap"] = pr["Price"] - pr["Competitor Prices"]
    pr["Suggested_Action"] = np.where(
        pr["Elasticity Index"] > 1.5,
        np.where(pr["Price_Gap"] > 0, "Lower Price", "Maintain"),
        "Raise Price"
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Price", f"${pr['Price'].mean():.2f}")
    col2.metric("Avg Competitor Price", f"${pr['Competitor Prices'].mean():.2f}")
    col3.metric("Avg Elasticity Index", f"{pr['Elasticity Index'].mean():.2f}")

    st.subheader("Recommended Pricing Actions")
    action_filter = st.multiselect(
        "Filter by action",
        options=pr["Suggested_Action"].unique().tolist(),
        default=pr["Suggested_Action"].unique().tolist()
    )
    st.dataframe(
        pr[pr["Suggested_Action"].isin(action_filter)]
        [["Product ID", "Store ID", "Price", "Competitor Prices", "Elasticity Index", "Suggested_Action"]]
        .head(50)
    )

    st.subheader("Action Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    pr["Suggested_Action"].value_counts().plot(kind="bar", color="coral", ax=ax)
    ax.set_ylabel("Number of Products")
    st.pyplot(fig)

    if models_loaded:
        st.subheader("Predict Sales Volume for a Given Price")
        c1, c2 = st.columns(2)
        with c1:
            p_price = st.number_input("Price ($)", min_value=0.0, value=25.0)
            p_comp = st.number_input("Competitor Price ($)", min_value=0.0, value=27.0)
            p_disc = st.number_input("Discount (%)", min_value=0.0, value=10.0)
        with c2:
            p_reviews = st.number_input("Customer Reviews (rating)", min_value=0.0, max_value=5.0, value=4.0)
            p_return = st.number_input("Return Rate (%)", min_value=0.0, value=5.0)
            p_storage = st.number_input("Storage Cost ($)", min_value=0.0, value=5.0)

        if st.button("Predict Sales Volume"):
            input_df = pd.DataFrame([{
                "Price": p_price,
                "Competitor Prices": p_comp,
                "Discounts": p_disc,
                "Customer Reviews": p_reviews,
                "Return Rate (%)": p_return,
                "Storage Cost": p_storage
            }])
            vol_pred = pricing_model.predict(input_df)[0]
            st.success(f"Predicted Sales Volume: **{vol_pred:.0f} units**")


st.sidebar.markdown("---")
st.sidebar.caption("Smart Inventory Intelligence · Built with Streamlit")