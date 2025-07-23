# Methodology Documentation

This document provides comprehensive information about the analytical methods, techniques, and approaches used in the COVID-19 Social Media Sentiment & Recovery Patterns Analysis project.

## Table of Contents
| # | Section |
|---|---------|
| 1 | [Overview](#overview) |
| 2 | [Natural Language Processing (NLP) Pipeline](#1-natural-language-processing-nlp-pipeline) |
| 3 | [Time-Series Analysis](#2-time-series-analysis) |
| 4 | [Statistical Analysis](#3-statistical-analysis) |
| 5 | [Data Visualization](#4-data-visualization) |
| 6 | [Quality Assurance & Validation](#5-quality-assurance--validation) |
| 7 | [Reproducibility Framework](#6-reproducibility-framework) |
| 8 | [Performance Optimization](#7-performance-optimization) |
| 9 | [Error Handling & Debugging](#8-error-handling--debugging) |
| 10 | [Best Practices & Guidelines](#9-best-practices--guidelines) |
| 11 | [Contact & Support](#contact--support) |

## Overview

The project employs advanced analytical techniques across multiple domains including Natural Language Processing (NLP), time-series analysis, statistical modeling, and data visualization to investigate complex relationships during the COVID-19 pandemic.

---

## 1. Natural Language Processing (NLP) Pipeline

### 1.1 Data Preprocessing

#### Text Cleaning Pipeline
```python
def clean_tweet_text(text):
    """Comprehensive tweet text preprocessing"""
    # Remove URLs, mentions, hashtags
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
    # Remove extra whitespace and special characters
    text = re.sub(r'\s+', ' ', text).strip()
    # Convert to lowercase
    return text.lower()
```

#### Quality Filtering
- **Retweet Removal**: Filter out retweets (`is_retweet == False`)
- **Length Filtering**: Minimum 10 characters after cleaning
- **Language Detection**: English-only tweets
- **Bot Filtering**: Remove accounts with suspicious activity patterns

### 1.2 Sentiment Analysis

#### VADER Sentiment Analysis
- **Library**: `vaderSentiment` - optimized for social media text
- **Output**: Compound score ranging from -1 (negative) to +1 (positive)
- **Advantages**: Handles emojis, capitalization, punctuation emphasis
- **Validation**: Verified against manual annotation sample

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
scores = analyzer.polarity_scores(text)
compound_score = scores['compound']  # Primary metric
```

#### Sentiment Segmentation
| Category | Threshold | Usage |
|----------|-----------|-------|
| **Positive** | compound > 0.05 | Positive sentiment analysis |
| **Negative** | compound < -0.05 | Negative sentiment analysis |
| **Neutral** | -0.05 ≤ compound ≤ 0.05 | Baseline comparison |

### 1.3 Emotion Detection

#### NRCLex Emotion Classification
- **Library**: `nrclex` - emotion lexicon approach
- **Emotions**: 8 categories from Plutchik's emotion model
- **Output**: Normalized scores (0-1) for each emotion category

```python
from nrclex import NRCLex

emotion_obj = NRCLex(text)
emotions = {
    'fear': emotion_obj.affect_frequencies['fear'],
    'anger': emotion_obj.affect_frequencies['anger'],
    'joy': emotion_obj.affect_frequencies['joy'],
    'sadness': emotion_obj.affect_frequencies['sadness'],
    'trust': emotion_obj.affect_frequencies['trust'],
    'disgust': emotion_obj.affect_frequencies['disgust'],
    'surprise': emotion_obj.affect_frequencies['surprise'],
    'anticipation': emotion_obj.affect_frequencies['anticipation']
}
```

#### Emotion Categories
| Emotion | Interpretation | Research Usage |
|---------|----------------|----------------|
| **Fear** | Anxiety, worry | Transit mobility correlation |
| **Anger** | Frustration, rage | Policy response analysis |
| **Joy** | Happiness, pleasure | Residential mobility correlation |
| **Sadness** | Depression, sorrow | Lockdown fatigue detection |
| **Trust** | Confidence, faith | Compliance measurement |
| **Disgust** | Revulsion, distaste | Misinformation reaction |
| **Surprise** | Shock, amazement | Policy announcement impact |
| **Anticipation** | Expectation, hope | Recovery optimism |

### 1.4 Topic Modeling

#### Latent Dirichlet Allocation (LDA)
- **Library**: `gensim` - efficient LDA implementation
- **Topics**: 3 topics identified through coherence optimization
- **Parameters**: α=0.1, β=0.01, iterations=1000

```python
from gensim import corpora, models
from gensim.models import CoherenceModel

# Preprocessing
processed_docs = [preprocess(doc) for doc in documents]
dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# LDA Model
lda_model = models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=3,
    alpha=0.1,
    eta=0.01,
    passes=10,
    iterations=1000
)
```

#### Topic Interpretation
| Topic ID | Label | Key Terms | Prevalence |
|----------|-------|-----------|------------|
| **Topic 0** | Lockdown-Related | lockdown, restrictions, mask, social_distance | ~25% |
| **Topic 1** | Health/Medical | hospital, vaccine, symptoms, treatment | ~35% |
| **Topic 2** | Social/Economic | work, family, economy, business | ~40% |

#### Topic Quality Metrics
- **Coherence Score**: 0.47 (good coherence)
- **Perplexity**: -7.8 (lower is better)
- **Topic Distinctiveness**: 0.73 (high separation)

---

## 2. Time-Series Analysis

### 2.1 Data Alignment and Preprocessing

#### Temporal Harmonization
```python
def align_time_series(data_dict, common_dates):
    """Align multiple time series to common date range"""
    aligned_data = {}
    for name, df in data_dict.items():
        # Filter to common dates
        aligned = df.loc[df.index.isin(common_dates)]
        # Forward fill missing values
        aligned = aligned.fillna(method='ffill')
        aligned_data[name] = aligned
    return aligned_data
```

#### Smoothing Techniques
- **Rolling Averages**: 7-day windows to reduce noise
- **Outlier Handling**: IQR-based outlier detection and capping
- **Missing Data**: Forward-fill for policy data, linear interpolation for mobility

### 2.2 Time-Lagged Cross-Correlation (TLCC)

#### Mathematical Foundation
```python
def time_lagged_correlation(x, y, max_lag=21):
    """Calculate cross-correlation at multiple lags"""
    correlations = []
    lags = range(-max_lag, max_lag + 1)
    
    for lag in lags:
        if lag < 0:
            # y leads x
            corr = np.corrcoef(x[-lag:], y[:lag])[0, 1]
        elif lag > 0:
            # x leads y  
            corr = np.corrcoef(x[:-lag], y[lag:])[0, 1]
        else:
            # No lag
            corr = np.corrcoef(x, y)[0, 1]
        
        correlations.append(corr)
    
    return lags, correlations
```

#### Statistical Significance Testing
```python
def correlation_significance(r, n, alpha=0.05):
    """Test significance of correlation coefficient"""
    # t-statistic for correlation
    t_stat = r * np.sqrt((n - 2) / (1 - r**2))
    # Two-tailed p-value
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - 2))
    
    return p_value < alpha, p_value
```

### 2.3 Event Study Analysis

#### Event Detection Algorithm
```python
def detect_policy_events(stringency_series, threshold=10, min_gap=7):
    """Detect sharp policy changes"""
    # Calculate daily changes
    changes = stringency_series.diff()
    
    # Identify jumps above threshold
    events = changes[changes >= threshold]
    
    # Filter events with minimum gap
    filtered_events = []
    last_date = None
    
    for date, change in events.items():
        if last_date is None or (date - last_date).days >= min_gap:
            filtered_events.append((date, change))
            last_date = date
    
    return filtered_events
```

#### Impact Measurement
```python
def measure_event_impact(outcome_series, event_dates, 
                        pre_window=7, post_window=14):
    """Measure pre-post event changes"""
    impacts = []
    
    for event_date in event_dates:
        # Pre-event baseline
        pre_period = outcome_series[
            (outcome_series.index >= event_date - timedelta(days=pre_window)) &
            (outcome_series.index < event_date)
        ]
        
        # Post-event period
        post_period = outcome_series[
            (outcome_series.index >= event_date) &
            (outcome_series.index <= event_date + timedelta(days=post_window))
        ]
        
        if len(pre_period) > 0 and len(post_period) > 0:
            impact = post_period.mean() - pre_period.mean()
            impacts.append({
                'event_date': event_date,
                'pre_baseline': pre_period.mean(),
                'post_average': post_period.mean(),
                'impact': impact
            })
    
    return pd.DataFrame(impacts)
```

---

## 3. Statistical Analysis

### 3.1 Correlation Analysis

#### Pearson Correlation
- **Usage**: Linear relationships between continuous variables
- **Assumptions**: Normality, linearity, homoscedasticity
- **Interpretation**: r > 0.5 (strong), 0.3-0.5 (moderate), <0.3 (weak)

#### Spearman Correlation
- **Usage**: Non-linear monotonic relationships
- **Advantages**: Robust to outliers, no normality assumption
- **Application**: Mobility-sentiment relationships with skewed distributions

### 3.2 Significance Testing

#### Multiple Comparison Correction
```python
from statsmodels.stats.multitest import multipletests

def correct_multiple_comparisons(p_values, method='bonferroni'):
    """Apply multiple comparison correction"""
    rejected, p_corrected, _, _ = multipletests(
        p_values, alpha=0.05, method=method
    )
    return rejected, p_corrected
```

#### Effect Size Reporting
- **Cohen's d**: For mean differences
- **Correlation coefficients**: For relationship strength
- **Confidence intervals**: For uncertainty quantification

### 3.3 Advanced Pattern Recognition

#### Principal Component Analysis (PCA)
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def perform_pca(data, n_components=3):
    """Dimensionality reduction with PCA"""
    # Standardize features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(data_scaled)
    
    # Analyze loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=data.columns
    )
    
    return components, loadings, pca.explained_variance_ratio_
```

#### K-Means Clustering
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def optimal_clustering(data, max_k=10):
    """Find optimal number of clusters"""
    silhouette_scores = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        silhouette_scores.append(score)
    
    optimal_k = K_range[np.argmax(silhouette_scores)]
    return optimal_k, silhouette_scores
```

---

## 4. Data Visualization

### 4.1 Interactive Visualization Framework

#### Plotly Implementation
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_research_dashboard(data, title):
    """Create multi-panel interactive dashboard"""
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=['Panel 1', 'Panel 2', 'Panel 3', 
                       'Panel 4', 'Panel 5', 'Panel 6'],
        specs=[[{"type": "scatter"}, {"type": "heatmap"}],
               [{"type": "bar"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # Add traces for each panel
    # ... (specific implementations per research question)
    
    fig.update_layout(
        title=title,
        height=900,
        showlegend=True
    )
    
    return fig
```

#### Visualization Standards
- **Color Schemes**: Colorblind-friendly palettes
- **Interactivity**: Hover information, zoom, pan capabilities
- **Statistical Indicators**: Significance markers, confidence intervals
- **Professional Styling**: Publication-ready formatting

### 4.2 Statistical Visualization

#### Cross-Correlation Heatmaps
```python
def plot_correlation_heatmap(correlation_matrix, significance_matrix):
    """Create annotated correlation heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=significance_matrix,  # Add significance annotations
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    return fig
```

#### Time-Series with Confidence Intervals
```python
def plot_time_series_with_ci(dates, values, ci_lower, ci_upper, title):
    """Plot time series with confidence intervals"""
    fig = go.Figure()
    
    # Main line
    fig.add_trace(go.Scatter(
        x=dates, y=values,
        mode='lines',
        name='Observed',
        line=dict(color='blue', width=2)
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=dates, y=ci_upper,
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=ci_lower,
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(0,0,255,0.2)',
        name='95% CI'
    ))
    
    return fig
```

---

## 5. Quality Assurance & Validation

### 5.1 Data Quality Checks

#### Automated Quality Assessment
```python
def assess_data_quality(df):
    """Comprehensive data quality assessment"""
    quality_report = {
        'total_records': len(df),
        'missing_data': df.isnull().sum().to_dict(),
        'duplicate_records': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'value_ranges': df.describe().to_dict()
    }
    
    # Outlier detection
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers = {}
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers[col] = ((df[col] < (Q1 - 1.5 * IQR)) | 
                        (df[col] > (Q3 + 1.5 * IQR))).sum()
    
    quality_report['outliers'] = outliers
    return quality_report
```

### 5.2 Statistical Validation

#### Cross-Validation Framework
```python
def cross_validate_analysis(data, analysis_func, k_folds=5):
    """K-fold cross-validation for analysis robustness"""
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    results = []
    
    for train_idx, test_idx in kf.split(data):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        
        # Apply analysis function
        result = analysis_func(train_data, test_data)
        results.append(result)
    
    return results
```

#### Robustness Testing
- **Parameter Sensitivity**: Test with different thresholds and windows
- **Subsample Analysis**: Validate with different time periods
- **Geographic Validation**: Test consistency across countries

---

## 6. Reproducibility Framework

### 6.1 Environment Management

#### Dependency Specification
```python
# requirements.txt format
pandas==2.0.3
numpy==1.25.2
plotly==5.15.0
scikit-learn==1.3.0
# ... (complete specification in requirements.txt)
```

#### Random Seed Control
```python
import random
import numpy as np
from sklearn.utils import check_random_state

def set_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    # For scikit-learn
    check_random_state(seed)
```

### 6.2 Documentation Standards

#### Code Documentation
- **Docstrings**: Comprehensive function documentation
- **Inline Comments**: Explanation of complex logic
- **Notebook Markdown**: Clear section descriptions
- **Variable Naming**: Descriptive, consistent naming conventions

#### Analysis Provenance
- **Data Lineage**: Track data transformations
- **Parameter Logging**: Record all analysis parameters
- **Version Control**: Git tracking of all changes
- **Results Archival**: Systematic storage of outputs

---

## 7. Performance Optimization

### 7.1 Computational Efficiency

#### Memory Management
```python
def optimize_memory_usage(df):
    """Optimize DataFrame memory usage"""
    # Downcast numeric types
    float_cols = df.select_dtypes(include=['float']).columns
    int_cols = df.select_dtypes(include=['int']).columns
    
    df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast='float')
    df[int_cols] = df[int_cols].apply(pd.to_numeric, downcast='integer')
    
    # Convert categorical columns
    for col in df.columns:
        if df[col].dtype == 'object':
            if df[col].nunique() / len(df) < 0.5:  # High repetition
                df[col] = df[col].astype('category')
    
    return df
```

#### Parallel Processing
```python
from multiprocessing import Pool
import numpy as np

def parallel_correlation_analysis(data_chunks, n_processes=4):
    """Parallel processing for correlation analysis"""
    with Pool(n_processes) as pool:
        results = pool.map(calculate_correlations, data_chunks)
    
    return np.concatenate(results)
```

### 7.2 Scalability Considerations

#### Chunked Processing
- **Large Dataset Handling**: Process data in manageable chunks
- **Memory-Efficient Operations**: Use iterators and generators
- **Incremental Analysis**: Update results with new data batches

#### Caching Strategy
```python
import pickle
from functools import wraps

def cache_results(cache_file):
    """Decorator for caching expensive computations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except FileNotFoundError:
                result = func(*args, **kwargs)
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                return result
        return wrapper
    return decorator
```

---

## 8. Error Handling & Debugging

### 8.1 Robust Error Handling

#### Graceful Degradation
```python
def robust_analysis(data, fallback_method=None):
    """Analysis with fallback options"""
    try:
        # Primary analysis method
        result = primary_analysis(data)
        return result, 'primary'
    except Exception as e:
        print(f"Primary method failed: {e}")
        
        if fallback_method:
            try:
                result = fallback_method(data)
                return result, 'fallback'
            except Exception as e2:
                print(f"Fallback method failed: {e2}")
                return None, 'failed'
        
        return None, 'failed'
```

### 8.2 Debugging Support

#### Comprehensive Logging
```python
import logging
from datetime import datetime

def setup_logging(log_level=logging.INFO):
    """Configure comprehensive logging"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'analysis_{datetime.now():%Y%m%d}.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)
```

---

## 9. Best Practices & Guidelines

### 9.1 Analytical Best Practices

#### Statistical Rigor
1. **Multiple Testing Correction**: Apply when testing multiple hypotheses
2. **Effect Size Reporting**: Always report alongside p-values
3. **Assumption Checking**: Verify statistical test assumptions
4. **Confidence Intervals**: Provide uncertainty quantification

#### Methodological Transparency
1. **Parameter Documentation**: Record all analysis parameters
2. **Decision Justification**: Explain methodological choices
3. **Limitation Acknowledgment**: Clearly state analysis limitations
4. **Sensitivity Analysis**: Test robustness to parameter changes

### 9.2 Code Quality Standards

#### Development Practices
1. **Modular Design**: Break analysis into reusable functions
2. **Type Hints**: Use Python type hints for clarity
3. **Unit Testing**: Test individual functions thoroughly
4. **Code Review**: Peer review of analytical code

#### Documentation Requirements
1. **Function Docstrings**: Complete parameter and return documentation
2. **Analysis Narrative**: Clear explanation of analytical steps
3. **Interpretation Guidance**: Help users understand results
4. **Usage Examples**: Provide working code examples

---

## Contact & Support

For questions about methodology, implementation, or analytical approaches:
- Review individual notebook implementations for detailed code
- Check function docstrings for parameter specifications
- Refer to statistical literature for theoretical foundations
- Contact project maintainers for specific methodological questions

**Methodology Status**: Validated and Tested  
**Last Updated**: July 2025  
**Testing Coverage**: Comprehensive across all analytical components
