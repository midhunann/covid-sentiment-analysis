# Dataset Documentation

This document provides comprehensive information about the four primary datasets used in the COVID-19 Social Media Sentiment & Recovery Patterns Analysis project.

## Table of Contents
| # | Section |
|---|---------|
| 1 | [Overview](#overview) |
| 2 | [COVID-19 Tweets Dataset](#1-covid-19-tweets-dataset) |
| 3 | [Google Community Mobility Reports](#2-google-community-mobility-reports) |
| 4 | [Oxford COVID-19 Government Response Tracker](#3-oxford-covid-19-government-response-tracker) |
| 5 | [Johns Hopkins University CSSE COVID-19 Data](#4-johns-hopkins-university-csse-covid-19-data) |
| 6 | [Dataset Integration Strategy](#dataset-integration-strategy) |
| 7 | [Data Quality Assessment](#data-quality-assessment) |
| 8 | [Processing Requirements](#processing-requirements) |
| 9 | [Usage Guidelines](#usage-guidelines) |
| 10 | [Contact & Support](#contact--support) |

## Overview

The project integrates four diverse, high-quality datasets to analyze the complex relationships between public sentiment, mobility patterns, government policies, and epidemiological outcomes during the COVID-19 pandemic.

## 1. COVID-19 Tweets Dataset

### Source & Access
- **Source**: Kaggle - COVID-19 Tweets Collection
- **File Location**: `data/raw/covid19_tweets/covid19_tweets.csv`
- **Format**: CSV (Comma-separated values)
- **Size**: 375,824 tweets (~50MB+)

### Dataset Structure
```
Columns: 13 fields
├── user_name          # Twitter username
├── user_location      # User-provided location (40% missing)
├── user_description   # User bio/description
├── user_created       # Account creation date
├── user_followers     # Follower count
├── user_friends       # Following count
├── user_favourites    # Likes count
├── user_verified      # Verification status (True/False)
├── date              # Tweet timestamp
├── text              # Tweet content (main analysis field)
├── hashtags          # Hashtags used
├── source            # Twitter client/platform
└── is_retweet        # Retweet flag (True/False)
```

### Temporal Coverage
- **Date Range**: July 24, 2020 - August 19, 2020 (26 days)
- **Frequency**: Continuous tweet stream
- **Coverage**: Global, English-language tweets with #COVID19 hashtag

### Data Quality Assessment
| Metric | Value | Notes |
|--------|-------|-------|
| **Total Records** | 375,824 | High volume for robust analysis |
| **Missing user_location** | ~40% | Geolocation inference required |
| **Retweets** | ~15% | Filtered out for original content |
| **Unique Users** | ~200,000 | Diverse user base |
| **Text Quality** | High | Minimal spam/bot content |

### Usage in Analysis
- **Sentiment Analysis**: VADER sentiment scoring (-1 to +1)
- **Emotion Detection**: NRCLex 8-emotion classification
- **Topic Modeling**: LDA with 3 topics (lockdown-related, compliance, other)
- **Text Processing**: Tokenization, cleaning, feature extraction

---

## 2. Google COVID-19 Community Mobility Reports

### Source & Access
- **Source**: Google LLC - Community Mobility Reports
- **File Location**: `data/raw/google_mobility/Global_Mobility_Report.csv`
- **Format**: CSV (Comma-separated values)
- **Documentation**: [Google Mobility Reports](https://www.google.com/covid19/mobility/)

### Dataset Structure
```
Columns: 15 fields
Geographic Identifiers:
├── country_region_code           # ISO country code
├── country_region               # Country name
├── sub_region_1                # State/province (filtered out)
├── sub_region_2                # City/county (filtered out)
├── metro_area                  # Metropolitan area
├── iso_3166_2_code            # ISO subdivision code
├── census_fips_code           # US census code
├── place_id                   # Google Places ID
└── date                       # Date (YYYY-MM-DD)

Mobility Categories (% change from baseline):
├── retail_and_recreation_percent_change_from_baseline
├── grocery_and_pharmacy_percent_change_from_baseline
├── parks_percent_change_from_baseline
├── transit_stations_percent_change_from_baseline
├── workplaces_percent_change_from_baseline
└── residential_percent_change_from_baseline
```

### Temporal Coverage
- **Date Range**: February 15, 2020 - October 15, 2022
- **Frequency**: Daily measurements
- **Baseline**: Median value for the same day of week (Jan 3 - Feb 6, 2020)

### Geographic Coverage
- **Countries**: 135 countries globally
- **Analysis Subset**: 10 major countries for focused analysis
  - United States, United Kingdom, Canada, Australia
  - Germany, France, Italy, Spain, Brazil, India

### Mobility Categories Explained
| Category | Description | Interpretation |
|----------|-------------|----------------|
| **Retail & Recreation** | Shopping centers, restaurants, cafes, museums | Discretionary social activity |
| **Grocery & Pharmacy** | Essential businesses | Necessary daily activities |
| **Parks** | National/local parks, beaches, public gardens | Outdoor recreation |
| **Transit Stations** | Public transport hubs | Work/travel mobility |
| **Workplaces** | Places of work | Economic activity |
| **Residential** | Duration spent at home | Staying home behavior |

### Data Quality Assessment
| Metric | Value | Notes |
|--------|-------|-------|
| **Total Records** | 2.3M+ | Comprehensive global coverage |
| **Missing Data** | <5% | High data completeness |
| **Country Coverage** | 135 countries | Global representation |
| **Daily Frequency** | Consistent | Reliable time series |

---

## 3. Oxford COVID-19 Government Response Tracker (OxCGRT)

### Source & Access
- **Source**: University of Oxford - Blavatnik School of Government
- **File Location**: `data/raw/oxford_government_response/OxCGRT_compact_national_v1.csv`
- **Format**: CSV (Comma-separated values)
- **Documentation**: [OxCGRT GitHub](https://github.com/OxCGRT/covid-policy-tracker)

### Dataset Structure
```
Columns: 40+ fields
Geographic & Temporal:
├── CountryName         # Country name
├── CountryCode        # ISO 3-letter code
├── Date              # YYYYMMDD format
└── Jurisdiction      # National/subnational

Policy Indices (0-100 scale):
├── StringencyIndex_Average           # Overall policy strictness
├── GovernmentResponseIndex_Average   # Government response breadth
├── ContainmentHealthIndex_Average    # Health system response
└── EconomicSupportIndex             # Economic support measures

Individual Policy Indicators (0-4 scale + flags):
├── C1M_School closing               # School closure policies
├── C2M_Workplace closing           # Workplace closure policies
├── C3M_Cancel public events        # Public event restrictions
├── C4M_Restrictions on gatherings  # Gathering size limits
├── C5M_Close public transport      # Public transport closure
├── C6M_Stay at home requirements   # Stay-at-home orders
├── C7M_Restrictions on internal movement # Travel restrictions
├── C8EV_International travel controls    # Border controls
├── E1_Income support               # Income support programs
├── E2_Debt/contract relief         # Debt relief measures
├── H1_Public information campaigns # Information campaigns
├── H2_Testing policy               # Testing strategies
├── H3_Contact tracing             # Contact tracing programs
└── [Additional health & vaccine policies]
```

### Temporal Coverage
- **Date Range**: January 1, 2020 - Present (ongoing)
- **Frequency**: Daily updates
- **Analysis Period**: Aligned with mobility data (Feb 2020 - Aug 2020)

### Geographic Coverage
- **Countries**: 180+ countries and territories
- **Analysis Subset**: Same 10 major countries as mobility analysis

### Key Indices Explained
| Index | Scale | Description | Usage |
|-------|-------|-------------|-------|
| **Stringency Index** | 0-100 | Overall strictness of lockdown policies | Primary policy measure |
| **Economic Support** | 0-100 | Financial support to citizens/businesses | Policy mix analysis |
| **Containment Health** | 0-100 | Health system response measures | Health policy focus |
| **Government Response** | 0-100 | Breadth of government actions | Overall response gauge |

### Data Quality Assessment
| Metric | Value | Notes |
|--------|-------|-------|
| **Total Records** | 400K+ | Comprehensive policy tracking |
| **Missing Data** | Minimal | High-quality manual coding |
| **Update Frequency** | Daily | Real-time policy tracking |
| **Countries** | 180+ | Global coverage |

---

## 4. JHU CSSE COVID-19 Data Repository

### Source & Access
- **Source**: Johns Hopkins University Center for Systems Science and Engineering
- **Files**: 
  - `data/raw/jhu_csse/time_series_covid19_confirmed_global.csv`
  - `data/raw/jhu_csse/time_series_covid19_deaths_global.csv`
- **Format**: CSV (Wide format - columns as dates)
- **Documentation**: [JHU CSSE GitHub](https://github.com/CSSEGISandData/COVID-19)

### Dataset Structure
```
Columns: 1000+ (dates as columns)
Geographic Identifiers:
├── Province/State      # State/province (can be empty)
├── Country/Region     # Country name
├── Lat               # Latitude coordinate
└── Long              # Longitude coordinate

Daily Case/Death Counts:
├── 1/22/20           # January 22, 2020 (cumulative)
├── 1/23/20           # January 23, 2020 (cumulative)
├── ...               # Daily columns
└── [Latest Date]     # Most recent data (cumulative)
```

### Temporal Coverage
- **Date Range**: January 22, 2020 - March 9, 2023
- **Frequency**: Daily cumulative counts
- **Analysis Usage**: Daily new cases (calculated as differences)

### Geographic Coverage
- **Countries**: 190+ countries and territories
- **Subnational**: State/province level for some countries
- **Analysis Level**: Country-level aggregation

### Data Processing Requirements
| Task | Method | Purpose |
|------|--------|---------|
| **Daily Conversion** | `df.diff(axis=1)` | Convert cumulative to daily new cases |
| **Geographic Aggregation** | Group by country | Country-level totals |
| **Date Parsing** | Column header conversion | Standard datetime format |
| **Quality Control** | Remove negative values | Handle reporting corrections |

### Data Quality Assessment
| Metric | Value | Notes |
|--------|-------|-------|
| **Total Records** | 750K+ | Comprehensive case tracking |
| **Missing Data** | Rare | High data reliability |
| **Update Frequency** | Daily | Consistent reporting |
| **Coverage** | Global | 190+ countries |

---

## Processed Datasets

### 1. Daily Tweet Sentiment & Topics
- **File**: `data/processed/daily_tweet_sentiment_topics.csv`
- **Purpose**: Aggregated daily sentiment and topic metrics
- **Processing**: VADER sentiment + NRCLex emotions + LDA topics
- **Timespan**: 26 days (July-August 2020)

### Structure
```
Columns: 11 fields
├── date_only                          # Date (YYYY-MM-DD)
├── vader_compound                     # Daily average sentiment (-1 to +1)
├── emotion_fear, emotion_anger, ...   # 8 emotion categories (0-1)
├── topic_prevalence_lockdown_related  # Topic 0 prevalence (0-1)
└── topic_prevalence_other             # Topic 1 prevalence (0-1)
```

### 2. Individual Tweet Features
- **File**: `data/processed/tweets_with_nlp_features.csv`
- **Purpose**: Tweet-level NLP analysis results
- **Processing**: Individual tweet sentiment, emotions, topics
- **Records**: 162,827 processed tweets (after filtering)

---

## Dataset Integration Strategy

### Temporal Alignment
```
Master Timeline: July 24 - August 19, 2020 (26 days)
├── Tweets: Native daily aggregation
├── Mobility: Daily Google reports
├── Policy: Daily Oxford indices
└── Cases: Daily JHU case counts
```

### Geographic Harmonization
```
Primary Analysis Countries: 10 countries
├── United States  ↔  US  ↔  United States
├── United Kingdom ↔  GB  ↔  United Kingdom
├── Canada         ↔  CA  ↔  Canada
├── Australia      ↔  AU  ↔  Australia
├── Germany        ↔  DE  ↔  Germany
├── France         ↔  FR  ↔  France
├── Italy          ↔  IT  ↔  Italy
├── Spain          ↔  ES  ↔  Spain
├── Brazil         ↔  BR  ↔  Brazil
└── India          ↔  IN  ↔  India
```

### Missing Data Handling
| Dataset | Missing Data Strategy |
|---------|----------------------|
| **Tweets** | Forward-fill for temporal gaps |
| **Mobility** | Linear interpolation for short gaps |
| **Policy** | Forward-fill (policies persist) |
| **Cases** | Linear interpolation for reporting delays |

---

## Usage Guidelines

### Data Loading
```python
# Processed data (recommended)
sentiment_data = pd.read_csv('data/processed/daily_tweet_sentiment_topics.csv', 
                           index_col=0, parse_dates=True)

# Raw data (for custom processing)
tweets_raw = pd.read_csv('data/raw/covid19_tweets/covid19_tweets.csv')
mobility_raw = pd.read_csv('data/raw/google_mobility/Global_Mobility_Report.csv')
policy_raw = pd.read_csv('data/raw/oxford_government_response/OxCGRT_compact_national_v1.csv')
cases_raw = pd.read_csv('data/raw/jhu_csse/time_series_covid19_confirmed_global.csv')
```

### Quality Considerations
1. **Temporal Scope**: Limited to 26-day overlap period
2. **Geographic Coverage**: Focus on 10 major countries for robust analysis
3. **Sample Size**: 162,827 processed tweets provide statistical power
4. **Data Reliability**: All sources are authoritative and well-documented

### Citation Requirements
- **Tweets**: Cite Kaggle dataset and original Twitter Academic API
- **Mobility**: Cite Google LLC Community Mobility Reports
- **Policy**: Cite Oxford COVID-19 Government Response Tracker
- **Cases**: Cite Johns Hopkins University CSSE COVID-19 Data Repository

---

## Contact & Support

For questions about dataset usage, processing, or quality:
- Review notebook `01_Dataset_Understanding_and_Exploration.ipynb`
- Check individual research question notebooks for specific usage patterns
- Refer to original data provider documentation for authoritative information

**Last Updated**: July 2025  
**Data Coverage**: July 24 - August 19, 2020
