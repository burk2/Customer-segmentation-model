import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model/customer_segmentation_model.pkl")

# Define cluster labels
cluster_labels = {
    0: "High-Spending Loyal Customers",
    1: "New Low-Spending Buyers",
    2: "Frequent Moderate-Spenders"
}

# Set page config
st.set_page_config(page_title="Customer Segmentation", page_icon="🧠", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧠 Customer Segmentation App</h1>", unsafe_allow_html=True)
st.write("Upload a customer CSV file to predict their segments using your clustering model.")

# Upload file
uploaded_file = st.file_uploader("📤 Upload CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    try:
        features = data[['recency', 'purchase_frequency', 'avg_order_value', 'total_spend', 'purchase_duration']]
        predictions = model.predict(features)

        data['cluster'] = predictions
        data['segment'] = data['cluster'].map(cluster_labels)

        # Style DataFrame with colors for each segment
        def highlight_segment(row):
            color_map = {
                "High-Spending Loyal Customers": "#A5D6A7",  # Green
                "New Low-Spending Buyers": "#FFF59D",        # Yellow
                "Frequent Moderate-Spenders": "#90CAF9"      # Blue
            }
            color = color_map.get(row['segment'], "#FFFFFF")
            return ['background-color: {}'.format(color) if col == 'segment' else '' for col in row.index]

        st.success("✅ Segmentation complete! See results below:")
        styled_df = data.style.apply(highlight_segment, axis=1)
        st.dataframe(styled_df, use_container_width=True)

        # Download button
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Segmented Data", csv, "segmented_customers.csv", "text/csv")

    except KeyError:
        st.error("❌ Missing required columns. Please include: 'recency', 'purchase_frequency', 'avg_order_value', 'total_spend', 'purchase_duration'.")
