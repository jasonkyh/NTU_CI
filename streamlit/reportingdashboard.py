#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 20:13:13 2025

@author: jasonkhoo
"""


import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
from PIL import Image
from pathlib import Path

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

# --- Define a BASE_DIR relative to this script's location ---
# This ensures all paths work in Streamlit Cloud.
BASE_DIR = Path(__file__).parent

# --- Use an image for the page icon ---
icon_path = BASE_DIR / "asset" / "icon-logo.png"
try:
    page_icon = Image.open(icon_path)
except FileNotFoundError:
    page_icon = "🏨" # Fallback icon

st.set_page_config(
    page_title="Hotel Review Insights Dashboard",
    page_icon=page_icon, # Use the loaded image
    layout="wide"
)

# ==============================================================================
# DATA LOADING AND PREPARATION
# ==============================================================================
# --- Load the sentiment analysis data ---
@st.cache_data
def load_data(path):
    """
    Loads the ABSA results, converts date columns, and creates new
    'year', 'month', 'has_sentiment', and 'overall_sentiment' columns.
    """
    if not path.exists():
        st.error(f"Data file not found at: {path}")
        st.stop()

    df = pd.read_csv(path)
    
    # Convert date columns, coercing errors to NaT (Not a Time).
    df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
    df['stay_date'] = pd.to_datetime(df['stay_date'], errors='coerce')
    
    # Add year and month columns for filtering
    df['year'] = df['review_date'].dt.year
    df['month'] = df['review_date'].dt.month

    # Create 'has_sentiment' Column
    aspect_sentiment_cols = [
        "Room Quality & Comfort_sentiment", "Facilities & Amenities_sentiment", 
        "Customer Service & Staff_sentiment", "Dining Experience_sentiment", 
        "Casino Experience_sentiment", "Shopping & Retail_sentiment", 
        "Location & Accessibility_sentiment", "Pricing & Value for Money_sentiment", 
        "Atmosphere & Ambience_sentiment"
    ]
    sentiment_cols = [col for col in aspect_sentiment_cols if col in df.columns]

    df['sentiment_count'] = df[sentiment_cols].notna().sum(axis=1)
    df['has_sentiment'] = np.where(df['sentiment_count'] > 0, "Yes", "No")

    # Calculate Overall Sentiment for each Review
    df['positive_aspects'] = df[sentiment_cols].apply(lambda row: (row == 'Positive').sum(), axis=1)
    df['negative_aspects'] = df[sentiment_cols].apply(lambda row: (row == 'Negative').sum(), axis=1)
    
    conditions = [
        (df['positive_aspects'] > df['negative_aspects']),
        (df['negative_aspects'] > df['positive_aspects']),
        ((df['positive_aspects'] == df['negative_aspects']) & (df['positive_aspects'] > 0))
    ]
    choices = ['Positive', 'Negative', 'Mixed']
    
    df['overall_sentiment'] = np.select(conditions, choices, default='Neutral/Not Analyzed')
    
    return df

# --- Load the AI-generated insights data ---
@st.cache_data
def load_insights(path):
    """Loads the CSV file containing AI-generated summaries and action plans."""
    if not path.exists():
        return None
    return pd.read_csv(path)

# --- Required for streamlit cloud deployment: load the data using the new, robust paths ---
data_file_path = BASE_DIR / "data" / "absa_analysis_twostep_results_wide.csv"
insights_file_path = BASE_DIR / "data" / "gemini_actionable_insights.csv"

# Load the data using the cached function
df = load_data(data_file_path)
insights_df = load_insights(insights_file_path)

# ==============================================================================
# SIDEBAR FILTERS
# ==============================================================================
st.sidebar.header("Filters")

# --- Add a Reset Button using Session State ---
def reset_filters():
    st.session_state.year_filter = 'Select All'
    st.session_state.month_filter = 'Select All'
    st.session_state.source_filter = 'Select All'
    st.session_state.traveler_type_filter = 'Select All'
    st.session_state.room_category_filter = 'Select All'
    st.session_state.room_view_filter = 'Select All'

if 'year_filter' not in st.session_state:
    reset_filters()

st.sidebar.button("Reset All Filters", on_click=reset_filters)
st.sidebar.markdown("---")

# Get unique sorted years and months for the dropdowns
years = ['Select All'] + sorted(df['year'].dropna().unique().astype(int), reverse=True)
months_map = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
available_months = sorted(df['month'].dropna().unique())
month_names = ['Select All'] + [months_map[m] for m in available_months]

# Create the selectbox widgets and link them to session state.
st.sidebar.selectbox("Select Year", options=years, key='year_filter')
st.sidebar.selectbox("Select Month", options=month_names, key='month_filter')

def create_selectbox_filter(label, column_name, key):
    options = ['Select All'] + list(df[column_name].dropna().unique())
    return st.sidebar.selectbox(label, options=options, key=key)

create_selectbox_filter("Source Platform", "source", 'source_filter')
create_selectbox_filter("Traveler Type", "traveler_type", 'traveler_type_filter')
create_selectbox_filter("Room Category", "room_category", 'room_category_filter')
create_selectbox_filter("Room View", "room_view", 'room_view_filter')

# --- Apply Filters to the DataFrame ---
df_filtered = df.copy()

if st.session_state.year_filter != 'Select All':
    df_filtered = df_filtered[df_filtered['year'] == st.session_state.year_filter]
if st.session_state.month_filter != 'Select All':
    month_num = list(months_map.keys())[list(months_map.values()).index(st.session_state.month_filter)]
    df_filtered = df_filtered[df_filtered['month'] == month_num]
if st.session_state.source_filter != 'Select All':
    df_filtered = df_filtered[df_filtered['source'] == st.session_state.source_filter]
if st.session_state.traveler_type_filter != 'Select All':
    df_filtered = df_filtered[df_filtered['traveler_type'] == st.session_state.traveler_type_filter]
if st.session_state.room_category_filter != 'Select All':
    df_filtered = df_filtered[df_filtered['room_category'] == st.session_state.room_category_filter]
if st.session_state.room_view_filter != 'Select All':
    df_filtered = df_filtered[df_filtered['room_view'] == st.session_state.room_view_filter]

# ==============================================================================
# MAIN DASHBOARD AREA
# ==============================================================================
header_logo_path = BASE_DIR / "asset" / "header-logo.jpg"
if header_logo_path.exists():
    st.image(str(header_logo_path))

st.title("Hotel Review Insights Dashboard")
st.markdown("An interactive dashboard for analyzing aspect-based sentiment from guest reviews.")

# --- KPI Scorecards (Top Section) ---
st.header("Overall Performance Snapshot")
df_analyzed = df_filtered.query("has_sentiment == 'Yes'")
unanalyzed_count = len(df_filtered.query("has_sentiment == 'No'"))
total_reviews = len(df_filtered)
total_analyzed = len(df_analyzed)
if total_analyzed > 0:
    positive_pct = (df_analyzed['overall_sentiment'] == 'Positive').sum() / total_analyzed * 100
    negative_pct = (df_analyzed['overall_sentiment'] == 'Negative').sum() / total_analyzed * 100
    mixed_neutral_pct = 100 - positive_pct - negative_pct
else:
    positive_pct, negative_pct, mixed_neutral_pct = 0, 0, 0
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reviews (Filtered)", f"{total_reviews:,}")
col2.metric("Positive Reviews", f"{positive_pct:.1f}%")
col3.metric("Negative Reviews", f"{negative_pct:.1f}%")
col4.metric("Mixed / Neutral Reviews", f"{mixed_neutral_pct:.1f}%")
st.info(f"ℹ️ **{total_analyzed:,} reviews with sentiment** were analyzed. **{unanalyzed_count:,} reviews** are not included (due to having <=5 words or no strong aspect relevance).")
st.markdown("---")

# --- Aspect Sentiment Ranking Chart ---
st.header("Aspect Sentiment Breakdown")
aspect_sentiment_cols = ["Room Quality & Comfort_sentiment", "Facilities & Amenities_sentiment", "Customer Service & Staff_sentiment", "Dining Experience_sentiment", "Casino Experience_sentiment", "Shopping & Retail_sentiment", "Location & Accessibility_sentiment", "Pricing & Value for Money_sentiment", "Atmosphere & Ambience_sentiment"]
sentiment_cols = [col for col in aspect_sentiment_cols if col in df.columns]
aspect_names = [col.replace('_sentiment', '') for col in sentiment_cols]

df_long_sentiment = df_analyzed.melt(
    id_vars=['review_id', 'review_date'],
    value_vars=sentiment_cols,
    var_name='aspect',
    value_name='sentiment'
).dropna()

df_long_sentiment['aspect'] = df_long_sentiment['aspect'].str.replace('_sentiment', '')
aspect_counts = df_long_sentiment.groupby(['aspect', 'sentiment']).size().reset_index(name='count')

if not aspect_counts.empty:
    aspect_totals = aspect_counts.groupby('aspect')['count'].sum().reset_index(name='total_mentions')
    aspect_counts = pd.merge(aspect_counts, aspect_totals, on='aspect')
    aspect_counts['Percentage'] = (aspect_counts['count'] / aspect_counts['total_mentions'])
    aspect_counts.rename(columns={'sentiment': 'Sentiment', 'count': 'Number of Mentions'}, inplace=True)
    fig_aspects = px.bar(
        aspect_counts, x='Number of Mentions', y='aspect', color='Sentiment', orientation='h',
        title="Positive vs. Negative Mentions for Each Aspect",
        color_discrete_map={'Positive': '#2ca02c', 'Negative': '#d62728', 'Neutral': '#7f7f7f'},
        labels={'aspect': 'Aspect'},
        category_orders={"Sentiment": ["Positive", "Negative", "Neutral"]},
        hover_name='aspect',
        hover_data={'Sentiment': True, 'Number of Mentions': True, 'Percentage': ':.1%', 'aspect': False}
    )
    fig_aspects.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_aspects, use_container_width=True)
else:
    st.warning("No sentiment data available for the selected filters.")

# --- Sentiment Trend Over Time ---
st.markdown("---")
st.header("Sentiment Trend Over Time")

# Widget to select an aspect to view its trend
selected_aspect_trend = st.selectbox("Select an Aspect to Track", options=aspect_names)

if selected_aspect_trend:
    # Filter the long-format data for the selected aspect
    trend_data = df_long_sentiment[df_long_sentiment['aspect'] == selected_aspect_trend]
    # Create a 'review_month' column by formatting the date
    trend_data['review_month'] = trend_data['review_date'].dt.strftime('%Y-%m')
    
    # Pivot the data to get monthly counts for each sentiment
    monthly_summary = trend_data.groupby(['review_month', 'sentiment']).size().unstack(fill_value=0)
    
    # Ensure all three sentiment columns exist
    for sentiment in ['Positive', 'Negative', 'Neutral']:
        if sentiment not in monthly_summary.columns:
            monthly_summary[sentiment] = 0
            
    # Calculate totals and percentages
    monthly_summary['Total Mentions'] = monthly_summary.sum(axis=1)
    monthly_summary['Positive %'] = (monthly_summary['Positive'] / monthly_summary['Total Mentions']) * 100
    monthly_summary['Negative %'] = (monthly_summary['Negative'] / monthly_summary['Total Mentions']) * 100

    # Prepare the final display DataFrame
    display_df = monthly_summary[['Total Mentions', 'Positive', 'Positive %', 'Negative', 'Negative %']].reset_index()
    display_df.rename(columns={'review_month': 'Month'}, inplace=True)
    
    # --- FIX 1: Sort by Month in ascending (chronological) order ---
    display_df = display_df.sort_values(by='Month', ascending=True)
    
    # Styling functions for the table
    def style_negative(v, props=''):
        return props if v > 0 else None
    def style_positive(v, props=''):
        return props if v > 0 else None

    if not display_df.empty:
        # --- FIX 2: Add use_container_width=True to make the table full-width ---
        st.dataframe(
            display_df.style
            .format({
                'Positive %': '{:.1f}%',
                'Negative %': '{:.1f}%',
            })
            .applymap(style_negative, props='color:red;', subset=['Negative', 'Negative %'])
            .applymap(style_positive, props='color:green;', subset=['Positive', 'Positive %']),
            use_container_width=True
        )
    else:
        st.warning(f"No data available for '{selected_aspect_trend}' in the selected period.")


# --- Deep Dive by Aspect ---
st.markdown("---")
st.header("Deep Dive by Aspect")
selected_aspect = st.selectbox("Select an Aspect to Analyze", options=aspect_names)
if selected_aspect and not df_analyzed.empty:
    aspect_specific_data = df_long_sentiment[df_long_sentiment['aspect'] == selected_aspect]
    total_mentions = len(aspect_specific_data)
    if total_mentions > 0:
        pos_mentions = (aspect_specific_data['sentiment'] == 'Positive').sum()
        neg_mentions = (aspect_specific_data['sentiment'] == 'Negative').sum()
        neu_mentions = total_mentions - pos_mentions - neg_mentions
        pos_pct = (pos_mentions / total_mentions) * 100
        neg_pct = (neg_mentions / total_mentions) * 100
        neu_pct = (neu_mentions / total_mentions) * 100
    else:
        pos_pct, neg_pct, neu_pct = 0, 0, 0
    st.subheader(f"Performance for: {selected_aspect}")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Mentions", f"{total_mentions:,}")
    kpi2.metric("Positive Mentions", f"{pos_pct:.1f}%")
    kpi3.metric("Negative Mentions", f"{neg_pct:.1f}%")
    kpi4.metric("Neutral Mentions", f"{neu_pct:.1f}%")
    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.subheader(f"Positive Reviews ({pos_mentions:,} mentions)")
        positive_review_ids = aspect_specific_data[aspect_specific_data['sentiment'] == 'Positive']['review_id']
        relevance_col = f"{selected_aspect}_relevance_score"
        sentiment_score_col = f"{selected_aspect}_sentiment_score"
        positive_reviews = df_filtered[df_filtered['review_id'].isin(positive_review_ids)].sort_values(by=[relevance_col, sentiment_score_col], ascending=[False, False])
        for index, row in positive_reviews.head(10).iterrows():
            with st.expander(f"⭐️ {row['review_score']} | {row['review_text'][:70]}..."):
                st.write(row['review_text'])
    with col_neg:
        st.subheader(f"Negative Reviews ({neg_mentions:,} mentions)")
        negative_review_ids = aspect_specific_data[aspect_specific_data['sentiment'] == 'Negative']['review_id']
        relevance_col = f"{selected_aspect}_relevance_score"
        sentiment_score_col = f"{selected_aspect}_sentiment_score"
        negative_reviews = df_filtered[df_filtered['review_id'].isin(negative_review_ids)].sort_values(by=[relevance_col, sentiment_score_col], ascending=[False, False])
        for index, row in negative_reviews.head(10).iterrows():
            with st.expander(f"⭐️ {row['review_score']} | {row['review_text'][:70]}..."):
                st.write(row['review_text'])
else:
    st.warning("Select an aspect to see a detailed breakdown.")


# --- Actionable Insights ---
st.markdown("---")
st.header("AI-Generated Actionable Insights")

if insights_df is not None:
    # Dropdown to select an aspect for which to see the AI summary
    insight_aspect = st.selectbox("Select an Aspect for AI Summary", options=aspect_names)
    
    if insight_aspect:
        # Filter the insights DataFrame for the selected aspect
        aspect_insight = insights_df[insights_df['aspect'] == insight_aspect]
        
        if not aspect_insight.empty:
            # --- FIX: Use separate column definitions for each "row" to ensure alignment ---

            # --- First Row: Analysis ---
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Positive Feedback Analysis")
                st.markdown(aspect_insight.iloc[0]['positive_sentiments_analysis'])
            with col2:
                st.subheader("Negative Feedback Analysis")
                st.markdown(aspect_insight.iloc[0]['negative_sentiments_analysis'])

            # Add a small visual separator for clarity
            st.markdown("<br>", unsafe_allow_html=True)

            # --- Second Row: Recommendations ---
            # Using a new set of columns ensures this row starts at the same vertical level.
            col3, col4 = st.columns(2)
            with col3:
                st.subheader("Recommendations to Maintain Strengths")
                st.markdown(aspect_insight.iloc[0]['positive_sentiments_action'])
            with col4:
                st.subheader("Recommendations for Improvement")
                st.markdown(aspect_insight.iloc[0]['negative_sentiments_action'])
        else:
            st.warning(f"No AI-generated insights available for '{insight_aspect}'.")
else:
    st.warning("Could not load AI-generated insights file ('gemini_actionable_insights.csv').")


# --- Full Filtered Review Data ---
st.markdown("---")
st.header("Full Filtered Review Data")

# --- Create a 'Detected Aspects' column for the final table ---
def get_detected_aspects(row):
    detected = [name for name, col in zip(aspect_names, sentiment_cols) if pd.notna(row[col])]
    return ', '.join(detected) if detected else 'None'

df_filtered['Detected Aspects'] = df_filtered.apply(get_detected_aspects, axis=1)

# --- Expanders with a single, full-width data_editor ---
st.write("Displaying all reviews that match the current filter selection. Double-click a cell in the 'Detected Aspects' and 'review_text' column to see the full text.")
st.data_editor(
    df_filtered[[
        'review_date', 'reviewer_name', 'review_score', 
        'Detected Aspects', 'review_text'
    ]],
    column_config={
        "review_text": st.column_config.TextColumn(
            "Review Text",
            width="large" # Make the column wider
        ),
        # --- Format the 'review_date' column to show only the date ---
        "review_date": st.column_config.DateColumn(
            "Review Date",
            format="YYYY-MM-DD",
        ),
    },
    use_container_width=True,
    hide_index=True,
    disabled=True # Make the table read-only
)



# How to Run the Dashboard

# 1.  Save the code above as `reportingdashboard.py`.
# 2.  Make sure your data file `data/absa_analysis_results_wide.csv` is in the correct location.
# 3.  Open your terminal, navigate to the folder where you saved the file, and run this command:
#   cd "/Users/jasonkhoo/Documents/NTU MSIS/AY2526 S1/IN6299 Critical Inquiry/Polars/Critical_Inquiry_VH-03-02/streamlit"
#   streamlit run reportingdashboard.py
    

