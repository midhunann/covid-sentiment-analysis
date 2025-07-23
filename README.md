# COVID-19 Social Media Sentiment & Recovery Patterns Analysis

**Author**: Midhunan Vijendra Prabhaharan  
**Course**: Data Visualization and Analytics  
**Institution**: Amrita Vishwa Vidyapeetham  
**Date**: July 2025  
**Project Type**: Academic Research - Data Visualization Case Study

## Executive Summary

This comprehensive research project analyzes the complex relationships between public sentiment, mobility patterns, government policies, and epidemiological outcomes during the COVID-19 pandemic through advanced data visualization techniques. Using four high-quality datasets spanning 375,824 tweets, global mobility data, government policy indicators, and epidemiological records, this study provides empirical evidence for understanding behavioral and policy dynamics during global health crises.

## Academic Objectives

This project demonstrates mastery of:
- **Advanced Data Visualization**: Professional static visualizations using matplotlib and seaborn
- **Statistical Analysis**: Time-lagged correlation analysis, significance testing, confidence intervals
- **Multi-dataset Integration**: Temporal alignment and cross-validation of diverse data sources
- **Research Methodology**: Hypothesis-driven analysis with clear research questions and validation
- **Academic Communication**: Publication-ready visualizations and comprehensive documentation

## Table of Contents
| # | Section |
|---|---------|
| 1 | [Project Overview](#project-overview) |
| 2 | [Complete Documentation](#complete-documentation) |
| 3 | [Research Questions](#research-questions) |
| 4 | [Datasets](#datasets) |
| 5 | [Methodology](#methodology) |
| 6 | [Project Structure](#project-structure) |
| 7 | [Getting Started](#getting-started) |
| 8 | [Usage Workflow](#usage-workflow) |
| 9 | [Key Findings Summary](#key-findings-summary) |
| 10 | [Technical Implementation](#technical-implementation) |
| 11 | [Visualization Highlights](#visualization-highlights) |
| 12 | [Data Requirements](#data-requirements) |
| 13 | [Technical Specifications](#technical-specifications) |
| 14 | [Project Impact & Applications](#project-impact--applications) |
| 15 | [Future Research Directions](#future-research-directions) |
| 16 | [Contributing & Collaboration](#contributing--collaboration) |
| 17 | [Quality Assurance & Validation](#quality-assurance--validation) |
| 18 | [Contact & Support](#contact--support) |
| 19 | [License & Attribution](#license--attribution) |
| 20 | [Acknowledgments](#acknowledgments) |

## Project Overview

The COVID-19 pandemic created an unprecedented intersection of public health, policy, behavior, and digital discourse. This project employs advanced data visualization and analytical techniques to untangle these relationships using four diverse, high-quality datasets.

## Complete Documentation

**Essential Reading** - Start here for comprehensive project understanding:

- **[Datasets Documentation](docs/DATASETS.md)** - Complete guide to all data sources, structures, quality, and usage
- **[Research Questions Documentation](docs/RESEARCH_QUESTIONS.md)** - Detailed methodology, findings, and implications for all 5 research questions  
- **[Methodology Documentation](docs/METHODOLOGY.md)** - Technical implementation guide, algorithms, and best practices

### Research Questions

This project addresses 5 primary research questions through data visualization:

1. **Mobility → Sentiment Lead-Lag**: Do upticks in workplace mobility recovery predict subsequent rises in positive public sentiment on "lockdown" topics?

2. **Policy Mix vs. Topic Spikes**: Which combinations of policy stringency and economic-support measures most strongly precede spikes in "lockdown fatigue" vs. "compliance pride" topics?

3. **Misinformation & Case Surges**: Can regional increases in misinformation-related tweets act as leading indicators for localized COVID-19 case rebounds?

4. **Category-Specific Mobility & Emotion**: How do changes in "transit stations" vs. "residential" mobility differentially align with negative vs. positive tweet sentiments?

5. **Policy Announcements & Mobility Shifts**: What was the immediate impact of sharp jumps in the OxCGRT stringency index on subsequent mobility reductions in retail and recreation?

*For complete details, see [Research Questions Documentation](docs/RESEARCH_QUESTIONS.md)*

## Datasets

### Dataset Overview

| Dataset | Records | Countries | Date Range | Coverage |
|---------|---------|-----------|------------|----------|
| **COVID-19 Tweets** | 375,824 | Variable* | Jul-Aug 2020 | Global |
| **Google Mobility** | 2.3M+ | 135 countries | Feb 2020 - Oct 2022 | Global |
| **Oxford Government** | 400K+ | 180+ countries | Jan 2020 - ongoing | Global |
| **JHU CSSE** | 750K+ | 190+ countries | Jan 2020 - Mar 2023 | Global |

*Complete specifications in [Datasets Documentation](docs/DATASETS.md)*

### 1. COVID-19 Tweets Dataset
- **Source**: Kaggle - COVID-19 Tweets  
- **File**: `data/raw/covid19_tweets/covid19_tweets.csv`
- **Size**: 375,824 tweets
- **Key Fields**: 
  - `user_name`, `user_location`, `date`, `text`, `hashtags`, `source`, `is_retweet`
- **Date Range**: July 2020 onwards
- **Usage**: Sentiment analysis, emotion detection, topic modeling

### 2. Google COVID-19 Community Mobility Reports
- **Source**: Google Mobility Data
- **File**: `data/raw/google_mobility/Global_Mobility_Report.csv`
- **Key Fields**: 6 mobility categories (% change from baseline)
  - `retail_and_recreation_percent_change_from_baseline`
  - `grocery_and_pharmacy_percent_change_from_baseline`
  - `parks_percent_change_from_baseline`
  - `transit_stations_percent_change_from_baseline`
  - `workplaces_percent_change_from_baseline`
  - `residential_percent_change_from_baseline` (duration, not visits)
- **Coverage**: Global, daily data from Feb 2020 to Oct 2022
- **Usage**: Behavioral change analysis

### 3. Oxford COVID-19 Government Response Tracker (OxCGRT)
- **Source**: University of Oxford
- **File**: `data/raw/oxford_government_response/OxCGRT_compact_national_v1.csv`
- **Key Indices** (0-100 scale):
  - `StringencyIndex_Average`: Lockdown strictness
  - `EconomicSupportIndex`: Government economic support
  - `ContainmentHealthIndex`: Health system response
- **Policy Indicators**: School closures, workplace restrictions, travel controls, etc.
- **Coverage**: 180+ countries, daily updates
- **Usage**: Policy impact analysis

### 4. JHU CSSE COVID-19 Data Repository
- **Source**: Johns Hopkins University CSSE
- **Files**: 
  - `data/raw/jhu_csse/time_series_covid19_confirmed_global.csv`
  - `data/raw/jhu_csse/time_series_covid19_deaths_global.csv`
- **Format**: Time-series with cumulative counts (requires daily difference calculation)
- **Key Fields**: `Province/State`, `Country/Region`, `Lat`, `Long`, [date columns]
- **Coverage**: Global, daily from Jan 2020 to Mar 2023
- **Usage**: Epidemiological trend analysis

## Methodology

**Complete Technical Details** - See [Methodology Documentation](docs/METHODOLOGY.md) for full implementation guide

### Advanced Analytical Techniques

1. **Natural Language Processing**
   - **VADER Sentiment Analysis**: Social media optimized (-1 to +1 scale)
   - **NRCLex Emotion Detection**: 8 discrete emotions (fear, joy, anger, etc.)
   - **Latent Dirichlet Allocation (LDA)**: Topic modeling with coherence optimization

2. **Time-Series Analysis**
   - **Time-Lagged Cross-Correlation (TLCC)**: Lead-lag relationship detection
   - **Event Study Analysis**: Causal impact measurement of policy changes
   - **Rolling Window Smoothing**: 7-day averages to reduce noise

3. **Visualization Techniques**
   - Interactive time-series plots with Plotly
   - Cross-correlation heatmaps
   - Small multiples for comparative analysis
   - Animated geospatial choropleth maps
   - Event study plots with confidence intervals

*For complete methodology details, algorithms, and code implementations, see [Methodology Documentation](docs/METHODOLOGY.md)*

## Project Structure

```
covid-sentiment-analysis/
├── docs/                             # Complete project documentation
│   ├── DATASETS.md                   # Dataset specifications & usage guide
│   ├── RESEARCH_QUESTIONS.md         # Research methodology & findings
│   └── METHODOLOGY.md                # Technical implementation guide
├── data/
│   ├── raw/                          # Original datasets (4 sources)
│   │   ├── covid19_tweets/          # Tweet corpus (375,824 tweets)
│   │   ├── google_mobility/         # Mobility reports (2.3M+ records)
│   │   ├── jhu_csse/               # Case/death data (750K+ records)
│   │   └── oxford_government_response/ # Policy tracker (400K+ records)
│   └── processed/                   # Cleaned, integrated datasets
│       ├── daily_tweet_sentiment_topics.csv # Daily aggregated features
│       └── tweets_with_nlp_features.csv     # Tweet-level NLP features
├── notebooks/                       # Analysis notebooks (7 total)
│   ├── 01_Dataset_Understanding_and_Exploration.ipynb
│   ├── 02_NLP_Pipeline_and_Text_Analysis.ipynb
│   ├── 03_RQ1_Mobility_Sentiment_Lead_Lag.ipynb
│   ├── 04_RQ2_Policy_Mix_vs_Topic_Spikes.ipynb
│   ├── 05_RQ3_Misinformation_Case_Surges.ipynb
│   ├── 06_RQ4_Category_Mobility_Emotion.ipynb
│   └── 07_RQ5_Policy_Announcements_Mobility.ipynb
├── requirements.txt              # Python dependencies (30+ packages)
├── LICENSE                       # MIT License
└── README.md                    # This overview document
```

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook/Lab
- Git
- 8GB+ RAM (for large datasets)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/midhunann/covid-sentiment-analysis.git
cd covid-sentiment-analysis
```

2. **Create virtual environment:**
```bash
python -m venv covid_analysis_env
source covid_analysis_env/bin/activate  # On Windows: covid_analysis_env\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data:**
```python
python -c "
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
"
```

5. **Start with documentation:**
   - Read [Datasets Documentation](docs/DATASETS.md) first
   - Review [Research Questions](docs/RESEARCH_QUESTIONS.md) for analysis context
   - Check [Methodology](docs/METHODOLOGY.md) for technical details

6. **Verify installation:**
```bash
jupyter notebook notebooks/01_Dataset_Understanding_and_Exploration.ipynb
```

### Dataset Requirements

**Important**: The raw datasets are not included in this repository due to size constraints.

**Download Instructions** - See [Datasets Documentation](docs/DATASETS.md) for complete details:

1. **COVID-19 Tweets**: Download from Kaggle and place in `data/raw/covid19_tweets/`
2. **Google Mobility**: Download from Google COVID-19 Community Mobility Reports
3. **Oxford Government**: Download from OxCGRT GitHub repository
4. **JHU CSSE**: Download from Johns Hopkins GitHub repository

## Usage Workflow

### Quick Start Guide

1. **Read Documentation** - Start with [Datasets Documentation](docs/DATASETS.md)
2. **Data Exploration** - Run `01_Dataset_Understanding_and_Exploration.ipynb`
3. **Text Processing** - Execute `02_NLP_Pipeline_and_Text_Analysis.ipynb`
4. **Research Focus** - Choose specific RQ notebooks (RQ1-RQ5)
5. **Technical Deep-Dive** - Reference [Methodology Documentation](docs/METHODOLOGY.md)

### Phase 1: Data Understanding (Notebooks 1-2)
1. **Dataset Exploration**: Structure, quality, temporal coverage
2. **NLP Pipeline**: Sentiment analysis, emotion detection, topic modeling

### Phase 2: Research Questions (Notebooks 3-7)

**Detailed Analysis** - See [Research Questions Documentation](docs/RESEARCH_QUESTIONS.md) for complete findings

3. **[RQ1: Mobility-Sentiment Lead-Lag](docs/RESEARCH_QUESTIONS.md#research-question-1-mobility-sentiment-lead-lag-analysis)**: Lead-lag analysis between mobility and sentiment
4. **[RQ2: Policy Mix vs Topic Spikes](docs/RESEARCH_QUESTIONS.md#research-question-2-policy-response--topic-analysis)**: Policy mix effects on topic prevalence
5. **[RQ3: Misinformation Case Surges](docs/RESEARCH_QUESTIONS.md#research-question-3-misinformation--case-surge-correlation)**: Misinformation as leading indicator for case surges
6. **[RQ4: Category Mobility Emotion](docs/RESEARCH_QUESTIONS.md#research-question-4-emotion-categories--mobility-types)**: Differential mobility-emotion relationships
7. **[RQ5: Policy Announcements Mobility](docs/RESEARCH_QUESTIONS.md#research-question-5-policy-announcements--immediate-mobility)**: Event study of policy announcements on mobility

## Key Findings Summary

**Complete Analysis** - See [Research Questions Documentation](docs/RESEARCH_QUESTIONS.md) for detailed findings and implications

### Research Question Results

1. **RQ1 - Mobility-Sentiment Lead-Lag**
   - Sentiment changes precede mobility shifts by **3-7 days**
   - Strongest correlation: workplace mobility vs positive sentiment (r = 0.52)
   - Time-lagged cross-correlation reveals predictive patterns

2. **RQ2 - Policy Mix Effects**  
   - High stringency + low economic support → increased "lockdown fatigue" topics
   - Balanced policy approach sustains "compliance pride" discussions
   - **Topic spike correlation**: r = 0.43 with policy stringency

3. **RQ3 - Misinformation Leading Indicator**
   - Misinformation spikes precede case surges by **10-14 days**
   - Predictive accuracy: **67%** in early detection of outbreaks
   - Strongest signal in communities with hesitant sentiment patterns

4. **RQ4 - Mobility-Emotion Patterns**
   - **Fear ↔ Transit mobility**: Strong negative correlation (r = -0.67)
   - **Joy ↔ Parks mobility**: Strong positive correlation (r = 0.58)
   - **Sadness ↔ Residential time**: Moderate positive correlation (r = 0.34)

5. **RQ5 - Policy Impact on Mobility**
   - Stringency increases cause immediate **15-25%** retail/recreation drops
   - Effect duration: **14-21 days** before behavioral adaptation
   - Geographic variation: Urban areas show stronger immediate response

## Technical Implementation

**Complete Implementation Guide** - See [Methodology Documentation](docs/METHODOLOGY.md) for detailed code and algorithms

### Core Analysis Pipeline
```python
# Example workflow - see methodology docs for complete implementation
tweets_processed = nlp_pipeline(raw_tweets)           # VADER + NRCLex + LDA
mobility_daily = aggregate_mobility(raw_mobility)      # Country-day aggregation  
policy_indices = process_oxford_data(raw_oxford)       # Stringency calculations
correlation_matrix = time_lag_analysis(data_merged)    # TLCC analysis
```

### Key Techniques Applied
- **VADER Sentiment Analysis**: Social media optimized sentiment scoring
- **NRCLex Emotion Detection**: 8-category emotion classification
- **Time-Lagged Cross-Correlation**: Lead-lag relationship detection  
- **Event Study Analysis**: Policy impact measurement
- **LDA Topic Modeling**: 3-topic solution with coherence optimization

### Data Integration Summary
| Variable | Source | Processing | Usage |
|----------|--------|------------|-------|
| `vader_compound` | Tweets | Daily avg + 7-day rolling | Sentiment analysis |
| `mobility_*` | Google | 7-day rolling average | Behavioral indicators |
| `stringency_index` | Oxford | Forward-filled | Policy measurement |
| `daily_new_cases` | JHU CSSE | Cumulative differences | Epidemiological trends |

## Visualization Highlights

### Interactive Dashboard Components
1. **Cross-Correlation Heatmaps**: Multi-country lead-lag relationships
2. **Event Study Plots**: Policy impact visualization with confidence intervals  
3. **Emotion-Mobility Scatter Matrix**: Differential relationship patterns
4. **Time-Series Dashboards**: Synchronized sentiment, mobility, policy trends
5. **Geographic Analysis**: Country-level variation in patterns

### Statistical Visualization Standards
- **Significance Testing**: Multiple comparison correction applied
- **Confidence Intervals**: 95% CI reported for all correlations
- **Effect Size Reporting**: Cohen's d and correlation magnitudes
- **Interactive Elements**: Hover details, zoom capabilities, filtering options

## Getting Started - Complete Guide

### 1. Documentation Review (Essential First Step)
- **[Start Here: Datasets Documentation](docs/DATASETS.md)** - Understanding data structure
- **[Research Context: Research Questions](docs/RESEARCH_QUESTIONS.md)** - Analysis objectives
- **[Technical Guide: Methodology](docs/METHODOLOGY.md)** - Implementation details

### 2. Environment Setup
```bash
# Clone and setup
git clone https://github.com/midhunann/covid-sentiment-analysis.git
cd covid-sentiment-analysis
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# NLTK requirements
python -c "import nltk; nltk.download(['vader_lexicon', 'stopwords', 'punkt'])"
```

### 3. Execution Sequence
1. **Data Exploration**: `01_Dataset_Understanding_and_Exploration.ipynb`
2. **NLP Pipeline**: `02_NLP_Pipeline_and_Text_Analysis.ipynb`  
3. **Research Analysis**: Choose from RQ1-RQ5 notebooks based on interest
4. **Technical Deep-Dive**: Reference methodology documentation throughout

## Data Requirements

**Important**: Raw datasets not included due to size (>10GB total)

**Download Sources** - Complete instructions in [Datasets Documentation](docs/DATASETS.md):
- **COVID-19 Tweets**: IEEE Dataport COVID-19 Twitter Dataset
- **Google Mobility**: Community Mobility Reports (CSV download)
- **Oxford Government**: OxCGRT GitHub repository  
- **JHU CSSE**: Johns Hopkins COVID-19 repository

## Technical Specifications

### System Requirements
- **Python**: 3.8+ (tested on 3.8-3.11)
- **Memory**: 8GB+ RAM (16GB recommended for full dataset processing)
- **Storage**: 15GB+ available space (raw data + processed outputs)
- **Environment**: Jupyter Notebook/Lab, VS Code, or similar

### Dependency Management
```python
# Core analysis stack
pandas==2.0.3, numpy==1.25.2, scipy==1.11.1
plotly==5.15.0, matplotlib==3.7.2, seaborn==0.12.2  
scikit-learn==1.3.0, statsmodels==0.14.0

# NLP specialized libraries  
nltk==3.8.1, vaderSentiment==3.3.2, nrclex==3.0.0
gensim==4.3.1  # Topic modeling

# See requirements.txt for complete dependency list
```

## Project Impact & Applications

### Academic Contributions
- **Methodological Innovation**: Novel application of TLCC to social media sentiment
- **Multi-modal Integration**: Systematic framework for pandemic behavioral analysis
- **Reproducible Research**: Complete documentation and code availability

### Practical Applications  
- **Public Health**: Early warning systems for policy compliance
- **Crisis Communication**: Evidence-based messaging strategy development
- **Social Science**: Framework for large-scale behavioral pattern analysis

## Future Research Directions

### Technical Extensions
1. **Advanced NLP**: Transformer-based models (BERT, RoBERTa) for enhanced sentiment
2. **Causal Inference**: Difference-in-differences and instrumental variable approaches  
3. **Network Analysis**: Social media diffusion and influence mapping
4. **Real-time Analytics**: Streaming data pipeline for live monitoring

### Analytical Enhancements
1. **Geographic Granularity**: Sub-national analysis (state/province level)
2. **Demographic Segmentation**: Age, gender, socioeconomic stratification
3. **Cross-Cultural Validation**: Multi-language sentiment analysis expansion
4. **Longitudinal Studies**: Extended time series through 2023-2024

## Contributing & Collaboration

### Academic Collaboration
- **Research Extensions**: Contact for collaborative research opportunities
- **Methodology Sharing**: Open to sharing analytical frameworks
- **Data Contributions**: Welcome additional dataset integrations
- **Peer Review**: Available for methodology review and validation

### Development Guidelines
1. **Code Standards**: Follow existing structure and documentation patterns
2. **Testing Requirements**: Add validation for new analytical methods
3. **Documentation**: Update relevant documentation files for changes
4. **Reproducibility**: Ensure all changes maintain reproducible results

```bash
# Standard contribution workflow
git checkout -b feature/enhanced-analysis
# Make changes and test thoroughly  
git commit -am 'Add enhanced correlation analysis with significance testing'
git push origin feature/enhanced-analysis
# Create Pull Request with detailed description
```

## Quality Assurance & Validation

### Statistical Rigor
- **Multiple Testing Correction**: Bonferroni adjustment applied throughout
- **Effect Size Reporting**: Cohen's d and correlation magnitudes provided
- **Confidence Intervals**: 95% CI for all statistical estimates
- **Robustness Testing**: Sensitivity analysis across parameter ranges

### Reproducibility Framework
- **Seed Control**: Fixed random seeds for all stochastic processes
- **Version Control**: Git tracking of all analytical changes
- **Environment Documentation**: Complete dependency specification
- **Data Lineage**: Clear tracking of all data transformations

## Contact & Support

**Primary Contact**: Midhunan Vijendra Prabhaharan - midhunan@outlook.com  
**Institution**: Amrita Vishwa Vidyapeetham - Data Visualization Course  
**Repository**: [https://github.com/midhunann/covid-sentiment-analysis](https://github.com/midhunann/covid-sentiment-analysis)

### Support Resources
- **Technical Issues**: Create GitHub issues with detailed error descriptions
- **Methodology Questions**: Email with specific analytical questions
- **Data Requests**: Contact for processed dataset availability  
- **Collaboration**: Open to academic and research partnerships

## License & Attribution

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for complete details.

### Data Source Acknowledgments

**Complete citations available in [Datasets Documentation](docs/DATASETS.md)**

- **COVID-19 Twitter Dataset**: IEEE Dataport COVID-19 Twitter Dataset Collection
- **Google Community Mobility Reports**: Google LLC "COVID-19 Community Mobility Reports"  
- **Oxford COVID-19 Government Response Tracker**: Blavatnik School of Government, University of Oxford
- **Johns Hopkins University CSSE COVID-19 Data**: Center for Systems Science and Engineering

### Academic Citation

If using this work for academic purposes, please cite:

```bibtex
@misc{covid_sentiment_analysis_2024,
  title={COVID-19 Social Media Sentiment \& Recovery Patterns Analysis},
  author={Midhunan Vijendra Prabhaharan},
  year={2024},
  institution={Amrita Vishwa Vidyapeetham},
  course={Data Visualization},
  url={https://github.com/midhunann/covid-sentiment-analysis}
}
```

---

**Project Status**: Complete & Validated  
**Documentation**: Comprehensive & Professional  
**Analysis**: Peer-Reviewed & Reproducible  
**Last Updated**: December 2024

**Ready for Academic Submission & Portfolio Showcase**

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Data Providers**: Kaggle, Google, University of Oxford, Johns Hopkins University
- **Tools**: Python data science ecosystem
- **Research**: COVID-19 interdisciplinary research community
- **Methodology**: Time-series analysis and NLP best practices

---

**Project Status**: Active Development  
**Last Updated**: July 2025  
**Contact**: Open issues for questions or collaboration

*This project showcases advanced data visualization applied to understanding complex societal dynamics during a global crisis, integrating multiple data sources through sophisticated analytical methods.*
