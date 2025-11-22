# VH-03-02 Analysis of Large Dataset using Polars Open-Source Library
#### Author: Khoo Yu Heng Jason (G2405638F), See Kwai Sheng (G2304088G), Tay Kian Koon (G2406555A)
*A Critical Inquiry Project submitted in partial fulfillment of the requirements for the Nanyang Technological University - Master of Science in Information Systems (MSIS) and Master of Science in Information Studies (IS).*

This repository contains the end-to-end data pipeline for a comprehensive analysis of hotel reviews. It scrapes, cleans, and consolidates reviews from 5 distinct sources, performs Aspect-Based Sentiment Analysis (ABSA), and generates AI-driven summaries, culminating in an interactive management dashboard built with Streamlit.

## Dashboard
The final interactive dashboard application is deployed on Streamlit Community Cloud.
### ➡️ **[View the Dashboard Here](https://ntu-ci-vh-03-02-absa-dashboard.streamlit.app/)**

## Core Features 
1. Scrapes data from 5 sources: Google Maps, Booking.com, Trip.com, Klook, and Trip Advisor.
2. Translates non-English reviews using the Google Cloud Translation API.
3. Consolidates & cleans all sources into a unified schema.
4. Benchmarks key data processing tasks (Consolidation, Imputation, Text Analysis) using Pandas vs. Polars.
5. Performs ABSA using a two-step model on a Google Colab T4 GPU.
6. Generates AI Summaries of positive/negative aspects using Gemini AI.
7. Visualizes all insights in an interactive Streamlit dashboard.

## Project Pipeline
The project is structured as a series of notebooks, designed to be run in the following order:
- 1A-1E (Web Scraping): Five separate notebooks using Selenium and Playwright to scrape raw review data from each source.
- 2 Translation.ipynb: Processes the scraped data and uses the Google Cloud Translation API to translate all non-English reviews to English for unified analysis.
- 3A Consolidation of Reviews.ipynb: Merges the five translated datasets, resolves schema differences, and creates a single, unified dataset. This notebook also contains the first benchmark for data consolidation.
- 3B EDA.ipynb: Performs comprehensive Exploratory Data Analysis, data imputation (handling missing values), and feature engineering. This notebook contains the core Pandas vs. Polars benchmarks for imputation and text analysis.
- 4A ABSA.ipynb: Implements a two-step Aspect-Based Sentiment Analysis pipeline to extract key aspects (e.g. "Room Quality & Comfort", "Customer Service & Staff") and their associated sentiment from review text.
- 4B AI Generated Summaries.ipynb: Uses the Google Gemini API to generate actionable, natural-language summaries from the ABSA results (e.g. "Top 20 positive reviews for "Dining Experience").
- reportingdashboard.py: A final Streamlit application that loads the ABSA and AI summary results to create an interactive dashboard for management insights.

## Repository Structure
The project is organized into logical directories reflecting the end-to-end data pipeline:

```text
.
├── Scripts/                                  # Core Jupyter Notebooks (Run in numerical order)
│   ├── 1A...1E. Web Scraping...ipynb         # Scraping scripts for 5 platforms (Selenium/Playwright)
│   ├── 2. Translation.ipynb                  # Google Cloud Translation API pipeline
│   ├── 3A. Consolidation of Reviews.ipynb    # Merging sources, schema alignment & cleaning
│   ├── 3B. EDA.ipynb                         # EDA, Imputation & Polars vs. Pandas Benchmarking
│   ├── 4A. ABSA.ipynb                        # Two-step Aspect-Based Sentiment Analysis pipeline
│   └── 4B. AI Generated Summaries.ipynb      # Gemini AI summarization & strategic recommendations
│
├── streamlit/                                # Interactive Dashboard Application
│   ├── reportingdashboard.py                 # Main Streamlit application script
│   ├── requirements.txt                      # App-specific dependencies
│   ├── data/                                 # Processed data subset required for the live app
│   └── asset/                                # Dashboard logos and icons
│
├── Benchmarking/                             # Performance Logs (Pandas vs. Polars)
│   ├── EDA_Benchmarking_...csv               # Runtime/Memory logs for Imputation & Text Analysis
│   └── Review_Consolidation_...csv           # Runtime/Memory logs for Data Consolidation
│
├── Output/                                   # Pipeline Outputs & Intermediate Files
│   ├── absa_analysis_...results.csv          # Final Sentiment Analysis results (Used in Dashboard)
│   ├── gemini_actionable_insights.csv        # Final AI Summaries (Used in Dashboard)
│   ├── cleaned_reviews_...csv                # Output from Step 3A (Consolidation)
│   └── imputed_reviews_...csv                # Output from Step 3B (Imputation)
│
├── Data/                                     # Source Data
│   ├── [platform]_reviews.json               # Raw scraped data (JSON/CSV)
│   └── [platform]...translated...csv         # Data processed via Translation API
│
└── README.md                                 # Project Documentation
