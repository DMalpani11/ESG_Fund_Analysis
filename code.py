import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression, VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import logging
import warnings


# Suppress warnings
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)

def load_data(file_path, date_columns=None):
    """
    Robust data loading with advanced datetime parsing
    
    :param file_path: Path to the CSV file
    :param date_columns: List of column names to parse as dates
    """
    try:
        # If date_columns is provided, use it for parsing
        if date_columns:
            raw_data = pd.read_csv(
                file_path, 
                low_memory=False,
                header=0,
                skiprows=1,
                parse_dates=date_columns
            )
        else:
            # Otherwise, use default parsing
            raw_data = pd.read_csv(
                file_path, 
                low_memory=False,
                header=0,
                skiprows=1,
                parse_dates=True
            )

        logging.info(f"Data loaded successfully. Shape: {raw_data.shape}")
        logging.info(f"Columns: {list(raw_data.columns)}")
    
        return raw_data
    
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logging.error("The CSV file is empty.")
        raise
    except Exception as e:
        logging.error(f"Unexpected error loading data: {e}")
        raise





def clean_data(raw_data):
    """
    Comprehensive data cleaning with advanced preprocessing
    """
    try:
        # Create a copy of raw data
        cleaned_data = raw_data.copy()

        
        
        # Identify numeric and categorical columns with non-null values
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
        categorical_columns = cleaned_data.select_dtypes(include=['object']).columns
        
        # Handle numeric columns
        if len(numeric_columns) > 0:
            # Remove columns with all NaN values in numeric columns
            numeric_columns = [col for col in numeric_columns if not cleaned_data[col].isna().all()]
            
            if numeric_columns:
                # Numeric imputation
                numeric_imputer = SimpleImputer(strategy='median')
                cleaned_data[numeric_columns] = numeric_imputer.fit_transform(
                    cleaned_data[numeric_columns]
                )
        
        # Handle categorical columns
        if len(categorical_columns) > 0:
            # Remove columns with all NaN values in categorical columns
            categorical_columns = [col for col in categorical_columns if not cleaned_data[col].isna().all()]
            
            if categorical_columns:
                # Categorical imputation
                categorical_imputer = SimpleImputer(strategy='most_frequent')
                cleaned_data[categorical_columns] = categorical_imputer.fit_transform(
                    cleaned_data[categorical_columns]
                )
        
        # Handle currency and percentage columns
        for col in cleaned_data.columns:
            if cleaned_data[col].dtype == 'object':
                # Remove currency symbols and commas
                cleaned_data[col] = (
                    cleaned_data[col]
                    .str.replace(r'[,$]', '', regex=True)
                    .str.replace('%', '')
                    .str.strip()
                )
                
                # Convert to numeric
                try:
                    cleaned_data[col] = pd.to_numeric(
                        cleaned_data[col], 
                        errors='coerce'
                    )
                except:
                    logging.warning(f"Could not convert column {col} to numeric")
        
        # Drop rows with all NaN values
        cleaned_data.dropna(how='all', inplace=True)
        
        # If no data remains after cleaning
        if cleaned_data.empty:
            logging.warning("All data was removed during cleaning process.")
            return None
        
        logging.info(f"Data cleaned successfully. Shape: {cleaned_data.shape}")
        return cleaned_data
    
    except Exception as e:
        logging.error(f"Error in data cleaning: {e}")
        return None




def enhanced_univariate_analysis(data):
    metrics = ['returns', 'risk', 'sharpe_ratio']
    metrics = [col for col in metrics if col in data.columns]  # Filter existing columns
    
    plt.figure(figsize=(15, 6))
    for i, col in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        # Drop NaN values for plotting
        sns.histplot(data[col].dropna(), kde=True, color='skyblue')
        plt.axvline(data[col].mean(), color='red', linestyle='--', label='Mean')
        plt.axvline(data[col].median(), color='green', linestyle='--', label='Median')
        plt.title(f'Distribution of {col.replace("_", " ").title()}')
        plt.xlabel(col.replace("_", " ").title())
        plt.ylabel('Frequency')
        plt.legend()
    plt.tight_layout()
    plt.show()

def enhanced_multivariate_analysis(data):
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    correlation = data[numeric_cols].corr().fillna(0)  # Handle NaN values in correlation matrix

    plt.figure(figsize=(120, 100))
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, mask=mask, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.xlabel('Features')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()

def perform_eda(data):
    """Perform univariate and multivariate analysis"""
    sns.set(style='whitegrid')  # Set seaborn style
    enhanced_univariate_analysis(data)
    enhanced_multivariate_analysis(data)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def preprocess_data(data):
    """Preprocess the dataset to handle missing values and prepare for analysis"""
    # Convert percentage strings to floats
    percentage_columns = data.select_dtypes(include=['object']).columns
    for col in percentage_columns:
        if data[col].str.contains('%').any():
            data[col] = data[col].str.rstrip('%').astype('float') / 100.0
            
    # Convert currency strings to floats
    currency_columns = data.filter(regex='asset|net assets').columns
    for col in currency_columns:
        data[col] = data[col].str.replace('$', '').str.replace(',', '').astype(float)
    
    return data

def analyze_fund_distribution(data):
    """Analyze the distribution of funds across different categories"""
    plt.figure(figsize=(15, 6))
    
    # Fund category distribution
    plt.subplot(1, 2, 1)
    category_counts = data['Fund profile: Category group'].value_counts()
    sns.barplot(x=category_counts.values, y=category_counts.index)
    plt.title('Distribution of Funds by Category')
    plt.xlabel('Number of Funds')
    
    # Asset manager distribution (top 10)
    plt.subplot(1, 2, 2)
    manager_counts = data['Fund profile: Asset manager'].value_counts().head(10)
    sns.barplot(x=manager_counts.values, y=manager_counts.index)
    plt.title('Top 10 Asset Managers by Number of Funds')
    plt.xlabel('Number of Funds')
    
    plt.tight_layout()
    plt.show()

def analyze_fund_performance(data):
    """Analyze fund performance metrics"""
    performance_periods = [
        'Returns and fees: Month end trailing returns, 1 year',
        'Returns and fees: Month end trailing returns, 3 year',
        'Returns and fees: Month end trailing returns, 5 year',
        'Returns and fees: Month end trailing returns, 10 year'
    ]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Performance distribution
    performance_data = data[performance_periods].melt()
    sns.boxplot(data=performance_data, x='variable', y='value', ax=ax1)
    ax1.set_title('Distribution of Returns Across Different Time Periods')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    
    # Expense ratio vs performance
    sns.scatterplot(
        data=data,
        x='Returns and fees: Prospectus net expense ratio',
        y='Returns and fees: Month end trailing returns, 5 year',
        ax=ax2
    )
    ax2.set_title('5-Year Returns vs Expense Ratio')
    
    plt.tight_layout()
    plt.show()

def analyze_esg_metrics(data):
    """Analyze ESG-specific metrics across the dataset"""
    # Create a 2x3 subplot grid to accommodate all metrics
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    
    # Environmental metrics
    env_metrics = [
        'Fossil Free Funds: Fossil fuel holdings, weight',
        'Fossil Free Funds: Clean200, weight',
        'Fossil Free Funds: Relative carbon footprint (tonnes CO2 / $1M USD invested)'
    ]
    
    # Create violin plots for environmental metrics
    env_data = data[env_metrics].melt()
    sns.violinplot(data=env_data, x='variable', y='value', ax=axes[0, 0])
    axes[0, 0].set_title('Distribution of Environmental Metrics')
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45)
    
    # Social metrics
    social_metrics = [
        'Gender Equality Funds: Gender equality score - Overall score (out of 100 points)',
        'Gun Free Funds: Civilian firearm, weight',
        'Prison Free Funds: All flagged, weight'
    ]
    
    social_data = data[social_metrics].melt()
    sns.violinplot(data=social_data, x='variable', y='value', ax=axes[0, 1])
    axes[0, 1].set_title('Distribution of Social Metrics')
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)
    
    # Deforestation metrics
    deforestation_metrics = [
        'Deforestation Free Funds: Deforestation-risk producer, weight',
        'Deforestation Free Funds: Deforestation-risk financier, weight',
        'Deforestation Free Funds: Deforestation-risk consumer brand, weight'
    ]
    
    deforestation_data = data[deforestation_metrics].melt()
    sns.violinplot(data=deforestation_data, x='variable', y='value', ax=axes[0, 2])
    axes[0, 2].set_title('Distribution of Deforestation Risk Metrics')
    axes[0, 2].set_xticklabels(axes[0, 2].get_xticklabels(), rotation=45)
    
    # Gender Equality Score vs 5-Year Returns
    sns.scatterplot(
        data=data,
        x='Gender Equality Funds: Gender equality score - Overall score (out of 100 points)',
        y='Returns and fees: Month end trailing returns, 5 year',
        ax=axes[1, 0]
    )
    axes[1, 0].set_title('Gender Equality Score vs 5-Year Returns')
    axes[1, 0].set_xlabel('Gender Equality Score')
    axes[1, 0].set_ylabel('5-Year Returns (%)')
    
    # Carbon Footprint vs Returns
    sns.scatterplot(
        data=data,
        x='Fossil Free Funds: Relative carbon footprint (tonnes CO2 / $1M USD invested)',
        y='Returns and fees: Month end trailing returns, 5 year',
        ax=axes[1, 1]
    )
    axes[1, 1].set_title('Carbon Footprint vs 5-Year Returns')
    axes[1, 1].set_xlabel('Carbon Footprint (tonnes CO2/$1M invested)')
    axes[1, 1].set_ylabel('5-Year Returns (%)')
    
    # Deforestation Risk vs Returns
    # Using total deforestation risk (sum of all three metrics)
    data['total_deforestation_risk'] = data[deforestation_metrics].sum(axis=1)
    sns.scatterplot(
        data=data,
        x='total_deforestation_risk',
        y='Returns and fees: Month end trailing returns, 5 year',
        ax=axes[1, 2]
    )
    axes[1, 2].set_title('Total Deforestation Risk vs 5-Year Returns')
    axes[1, 2].set_xlabel('Total Deforestation Risk Weight (%)')
    axes[1, 2].set_ylabel('5-Year Returns (%)')
    
    # Add a main title to the figure
    fig.suptitle('ESG Metrics Analysis: Environmental, Social, and Deforestation Impacts', 
                 fontsize=16, y=1.02)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
    
    # Additional analysis: Calculate correlations
    correlation_metrics = [
        'Returns and fees: Month end trailing returns, 5 year',
        'Gender Equality Funds: Gender equality score - Overall score (out of 100 points)',
        'Fossil Free Funds: Relative carbon footprint (tonnes CO2 / $1M USD invested)',
        'total_deforestation_risk'
    ]
    
    correlations = data[correlation_metrics].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations, annot=True, cmap='RdYlBu', center=0)
    plt.title('Correlation Between ESG Metrics and Returns')
    plt.tight_layout()
    plt.show()







def eda_analysis(data):
    
    
    
    
    print("\nanalyzing fund distribution...")
    analyze_fund_distribution(data)
    
    print("\nAnalyzing fund performance...")
    analyze_fund_performance(data)
    
    # Calculate summary statistics
    print("\nanalyzing esg metrics...")
    analyze_esg_metrics(data)
    
   

    

def analyze_returns(cleaned_data):
    """Comprehensive Returns Analysis"""
    return_columns = ['Returns', 'Return', 'Total Return', 'Annual Return',
                      'Yearly Return', 'Net Return', 'ROI', 'Return Rate',
                      'Month end trailing returns, year-to-date']
    
    # Filter columns that match return-related keywords
    potential_return_cols = [
        col for col in cleaned_data.columns
        if any(keyword.lower() in col.lower() for keyword in return_columns)
    ]
    
    # Log the number of potential return columns found
    logging.info(f"Potential return columns found: {len(potential_return_cols)}")
    
    # If no return columns found, attempt to use numeric columns that are between -100% and 100%
    if not potential_return_cols:
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        potential_return_cols = [
            col for col in numeric_cols
            if cleaned_data[col].between(-100, 100).any()
        ]
    
    if not potential_return_cols:
        logging.error("No suitable return columns found.")
        return None
    
    returns_analysis = {}
    all_returns = []  # List to hold all the returns for combined analysis

    for col in potential_return_cols:
        # Convert column to numeric, coerce errors to NaN
        numeric_returns = pd.to_numeric(cleaned_data[col], errors='coerce')

        # Skip column if all values are NaN
        if numeric_returns.isna().all():
            logging.warning(f"Skipping {col}, all values are NaN.")
            continue

        # Perform analysis
        col_analysis = {
            'mean_return': numeric_returns.mean(),
            'median_return': numeric_returns.median(),
            'return_volatility': numeric_returns.std(),
            'min_return': numeric_returns.min(),
            'max_return': numeric_returns.max(),
            'positive_return_percentage': (numeric_returns > 0).mean() * 100
        }
        returns_analysis[col] = col_analysis

        # Add returns to all_returns for combined risk analysis
        all_returns.extend(numeric_returns.dropna())

        # Plot histogram
        plt.figure(figsize=(10, 6))
        numeric_returns.hist(bins=30, color='skyblue', edgecolor='black')
        plt.title(f'Distribution of {col}')
        plt.xlabel('Returns')
        plt.ylabel('Frequency')

        # Adding mean and median lines
        plt.axvline(numeric_returns.mean(), color='red', linestyle='--', label=f'Mean: {numeric_returns.mean():.2f}')
        plt.axvline(numeric_returns.median(), color='green', linestyle='--', label=f'Median: {numeric_returns.median():.2f}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Add the aggregated 'returns' column to cleaned_data for further analysis
    cleaned_data['returns'] = pd.Series(all_returns)
    
    # Log the length of all returns
    logging.info(f"Total valid returns added: {len(all_returns)}")

    return returns_analysis

def calculate_risk_metrics(data):
    """
    Calculate common risk metrics for investment data
    """
    try:
        # Check if 'returns' column is present after the analysis
        if 'returns' not in data.columns:
            logging.error("No 'returns' column found in the dataset")
            return None

        # Calculate daily volatility (standard deviation of returns)
        volatility = data['returns'].std()
        
        # Calculate annualized volatility (assuming 252 trading days per year)
        annualized_volatility = volatility * np.sqrt(252)
        
        # Calculate the Sharpe ratio (assuming a risk-free rate of 0)
        risk_free_rate = 0.0  # This can be adjusted based on your context
        sharpe_ratio = (data['returns'].mean() - risk_free_rate) / volatility
        
        # Calculate Value at Risk (VaR) at 95% confidence level
        var_95 = np.percentile(data['returns'], 5)
        
        # Calculate Beta (assuming you have market returns in 'market_returns' column)
        if 'market_returns' in data.columns:
            covariance = np.cov(data['returns'], data['market_returns'])[0, 1]
            market_variance = data['market_returns'].var()
            beta = covariance / market_variance
        else:
            beta = np.nan  # If market returns are not provided, set beta as NaN
        
        # Compile results in a dictionary
        risk_metrics = {
            'volatility': volatility,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95,
            'beta': beta
        }
        
        # Log the calculated risk metrics
        logging.info(f"Risk metrics calculated: {risk_metrics}")
        
        # Add risk metrics as new columns to the data
        for key, value in risk_metrics.items():
            data[key] = value
        
        logging.info("Risk metrics added to the dataset successfully.")
        return data
    
    except Exception as e:
        logging.error(f"Error calculating risk metrics: {e}")
        return None


def create_esg_score(data):
    #Create composite ESG score
    # ESG-related keywords in column names
    esg_columns = [
        col for col in data.columns
        if any(keyword in col.lower() for keyword in ['environmental', 'social', 'governance', 'sustainability'])
    ]

    if not esg_columns:
        print("No ESG-related columns found.")
        return data

    # Normalize ESG columns and create composite ESG score
    for col in esg_columns:
        if data[col].dtype in [np.number]:
            data[f'{col}_normalized'] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

    # Create composite ESG score
    normalized_cols = [col for col in data.columns if col.endswith('_normalized')]
    if normalized_cols:
        data['composite_esg_score'] = data[normalized_cols].mean(axis=1)

    return data

def encode_categorical_features(data):
    """Encode categorical variables"""
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if data[col].nunique() < 10:
            encoded = pd.get_dummies(data[col], prefix=col)
            data = pd.concat([data, encoded], axis=1)
        else:
            data[f'{col}_encoded'] = pd.factorize(data[col])[0]
    return data

    



logging.basicConfig(level=logging.INFO)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import logging

logging.basicConfig(level=logging.INFO)

def prepare_return_metrics(data):
    """Extract and prepare return metrics from the dataset"""
    return_columns = [
        'Returns and fees: Month end trailing returns, 1 month',
        'Returns and fees: Month end trailing returns, 1 year',
        'Returns and fees: Month end trailing returns, 3 year',
        'Returns and fees: Month end trailing returns, 5 year'
    ]
    
    # Convert return columns to numeric
    for col in return_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Calculate average return across different timeframes
    data['avg_return'] = data[return_columns].mean(axis=1)
    
    return data

def calculate_esg_composite_score(data):
    """Calculate comprehensive ESG score based on multiple factors"""
    
    # Environmental factors
    env_factors = {
        'fossil_free': 1 - data['Fossil Free Funds: Fossil fuel holdings, weight'],
        'clean_energy': data['Fossil Free Funds: Clean200, weight'],
        'carbon_footprint': 1 / (data['Fossil Free Funds: Relative carbon footprint (tonnes CO2 / $1M USD invested)'] + 1)
    }
    
    # Social factors
    social_factors = {
        'gender_equality': data['Gender Equality Funds: Gender equality score - Overall score (out of 100 points)'] / 100,
        'weapon_free': 1 - data['Weapon Free Funds: Military weapon, weight'],
        'prison_free': 1 - data['Prison Free Funds: All flagged, weight']
    }
    
    # Governance factors (using available proxies)
    gov_factors = {
        'transparency': data['Gender Equality Funds: Gender equality score - Commitment, transparency, and accountability (out of 10 points)'] / 10
    }
    
    # Calculate composite scores
    data['environmental_score'] = pd.DataFrame(env_factors).mean(axis=1)
    data['social_score'] = pd.DataFrame(social_factors).mean(axis=1)
    data['governance_score'] = pd.DataFrame(gov_factors).mean(axis=1)
    
    # Calculate overall ESG score
    data['esg_composite_score'] = (
        data['environmental_score'] * 0.4 +
        data['social_score'] * 0.4 +
        data['governance_score'] * 0.2
    )
    
    return data

def analyze_risk_return_profile(data):
    """Analyze the relationship between ESG scores, returns, and risk"""
    
    # Calculate risk metrics
    data['return_volatility'] = data[[
        'Returns and fees: Month end trailing returns, 1 month',
        'Returns and fees: Month end trailing returns, 3 month',
        'Returns and fees: Month end trailing returns, 6 month',
        'Returns and fees: Month end trailing returns, 1 year'
    ]].std(axis=1)
    
    # Create fund classifications
    data['esg_category'] = pd.qcut(data['esg_composite_score'], 
                                 q=4, 
                                 labels=['Low ESG', 'Medium-Low ESG', 
                                       'Medium-High ESG', 'High ESG'])
    
    return data

def identify_key_factors(data):
    """Identify key factors influencing fund performance"""
    
    # Select relevant features for analysis
    features = [
        'environmental_score', 'social_score', 'governance_score',
        'Fossil Free Funds: Carbon footprint portfolio coverage by market value weight',
        'Gender Equality Funds: Gender equality score - Overall score (out of 100 points)',
        'Returns and fees: Prospectus net expense ratio',
        'Fund profile: Fund net assets',
        'return_volatility',
        'avg_return'
    ]
    
    # Prepare data for analysis
    analysis_data = data[features].copy()
    analysis_data = analysis_data.dropna()
    
    # Standardize the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(analysis_data)
    
    # Perform PCA to identify key components
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)
    
    # Calculate feature importance
    feature_importance = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(len(features))],
        index=features
    )
    
    return feature_importance

def plot_esg_performance_analysis(data):
    """Create visualizations for ESG performance analysis"""
    
    # Plot 1: ESG Score vs Returns
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=data, x='esg_composite_score', y='avg_return', 
                   hue='esg_category', alpha=0.6)
    plt.title('ESG Score vs Average Returns')
    plt.xlabel('ESG Composite Score')
    plt.ylabel('Average Return (%)')
    plt.show()
    
    # Plot 2: Risk-Return Profile by ESG Category
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x='esg_category', y='return_volatility')
    plt.title('Risk Profile by ESG Category')
    plt.xlabel('ESG Category')
    plt.ylabel('Return Volatility')
    plt.xticks(rotation=45)
    plt.show()

def main_analysis(data):
    """Main function to run the complete analysis"""
    
   
    
    # Prepare return metrics
    data = prepare_return_metrics(data)
    
    # Calculate ESG scores
    data = calculate_esg_composite_score(data)
    
    # Analyze risk-return profile
    data = analyze_risk_return_profile(data)
    
    # Identify key factors
    feature_importance = identify_key_factors(data)
    
    # Generate visualizations
    plot_esg_performance_analysis(data)
    
    # Identify top performing funds
    top_funds = data[
        (data['esg_composite_score'] > data['esg_composite_score'].quantile(0.75)) &
        (data['avg_return'] > data['avg_return'].quantile(0.75))
    ].sort_values('avg_return', ascending=False)
    
    return {
        'data': data,
        'feature_importance': feature_importance,
        'top_funds': top_funds[['Fund profile: Fund name', 'esg_composite_score', 
                               'avg_return', 'return_volatility']]
    }
"""{def detect_anomalies(data):
    #Detect anomalies using various methods and drop non-numeric anomalies.
    anomalies = {}

    # Filter only numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns  # Only include numeric columns

    # Z-score based anomaly detection
    for col in numeric_cols:
        # Check if the column has valid numeric data and is not empty
        if data[col].dropna().empty:
            logging.warning(f"Skipping {col}, as it contains no valid numeric data.")
            continue
        
        z_scores = zscore(data[col].dropna())
        # Find outliers based on Z-scores
        anomaly_data = data[abs(z_scores) > 3]
        if not anomaly_data.select_dtypes(include=[np.number]).empty:  # Retain only if numeric data exists
            anomalies[f'{col}_zscore_anomalies'] = anomaly_data

    # High performance, low cost anomaly detection
    if 'returns' in data.columns and 'expense_ratio' in data.columns:
        high_perf_low_cost = data[
            (data['returns'] > data['returns'].quantile(0.75)) & 
            (data['expense_ratio'] < data['expense_ratio'].quantile(0.25))
        ]
        if not high_perf_low_cost.empty:
            anomalies['high_performance_low_cost'] = high_perf_low_cost

    # ESG-return inconsistency anomaly detection
    if 'composite_esg_score' in data.columns and 'returns' in data.columns:
        esg_return_correlation = data[
            (data['composite_esg_score'] > data['composite_esg_score'].quantile(0.75)) & 
            (data['returns'] < data['returns'].quantile(0.25))
        ]
        if not esg_return_correlation.empty:
            anomalies['esg_return_inconsistency'] = esg_return_correlation

    # Drop non-numeric anomaly entries
    anomalies = {k: v for k, v in anomalies.items() if not v.select_dtypes(include=[np.number]).empty}

    return anomalies


def select_features(data, target_col='returns'):
    #Select most relevant features
    selector = VarianceThreshold(threshold=0.01)
    numeric_data = data.select_dtypes(include=[np.number])

    # Remove columns with too many missing values
    empty_cols = numeric_data.columns[numeric_data.isna().sum() == len(numeric_data)]
    logging.info(f"Columns with only missing values: {empty_cols}")
    numeric_data = numeric_data.drop(columns=empty_cols)

    # Check if there's more than one numeric column to process
    if len(numeric_data.columns) > 1:
        selected_features = selector.fit_transform(numeric_data)
        selected_features_df = pd.DataFrame(selected_features, columns=numeric_data.columns[selector.get_support()])

        if target_col in data.columns:
            mi_scores = mutual_info_regression(selected_features_df.fillna(0), data[target_col].fillna(0))
            feature_importance = pd.DataFrame({
                'feature': selected_features_df.columns,
                'importance': mi_scores
            }).sort_values('importance', ascending=False)

            return feature_importance

    return None

def visualize_anomalies(data, anomalies):
    #Visualize detected anomalies
    if not anomalies:
        logging.warning("No anomalies detected.")
        return
    
    for anomaly_type, anomaly_data in anomalies.items():
        if len(anomaly_data) > 0:
            plt.figure(figsize=(12, 6))  # Adjust figure size if needed

            # Example: plotting anomalies for returns vs expense ratio
            if 'returns' in data.columns and 'expense_ratio' in data.columns:
                # Check if the anomaly data is valid for plotting
                if not anomaly_data.empty:
                    plt.scatter(data['expense_ratio'], data['returns'], alpha=0.5, label='Normal')
                    plt.scatter(anomaly_data['expense_ratio'], anomaly_data['returns'], color='red', label='Anomaly')
                    plt.xlabel('Expense Ratio')
                    plt.ylabel('Returns')
                    plt.title(f'Anomalies: {anomaly_type}')
                    plt.legend()
                else:
                    logging.warning(f"No anomalies found for {anomaly_type}")
            
            plt.tight_layout()
            plt.show()
        else:
            logging.warning(f"No anomalies found for {anomaly_type}") """
class FundAnalysisModels:
    def __init__(self, data, target_col='returns', test_size=0.2, random_state=42):
        self.data = data
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
    def clean_data(self, data):
        """
        Pre-process data by removing columns with all missing values and reset index
        """
        # Remove columns where all values are missing
        non_empty_cols = data.columns[data.notna().any()]
        cleaned_data = data[non_empty_cols].copy()
        
        # Reset index to avoid any index-related shape mismatches
        cleaned_data = cleaned_data.reset_index(drop=True)
        
        # Convert all numeric columns to float, handling any string/object columns
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        cleaned_data[numeric_cols] = cleaned_data[numeric_cols].astype(float)
        
        return cleaned_data

    def prepare_data(self, data):
        # Clean the data first
        cleaned_data = self.clean_data(data)
        
        # Ensure target column exists
        if self.target_col not in cleaned_data.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in data")
            
        # Separate features and target
        X = cleaned_data.drop([self.target_col], axis=1).select_dtypes(include=[np.number])
        y = cleaned_data[self.target_col]
        
        # Log the shape of data for debugging
        print(f"Shape of X: {X.shape}")
        print(f"Shape of y: {y.shape}")
        
        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
        
        # Save feature names for later use
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert to DataFrame to maintain column names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

    def train_linear_regression(self, X_train, y_train):
        """
        Train a linear regression model and return feature importance
        """
        try:
            print("Training Linear Regression model...")
            model = LinearRegression()
            model.fit(X_train, y_train)
            self.models['linear'] = model

            # Calculate feature importance
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': abs(model.coef_)
            }).sort_values('importance', ascending=False)

            print("Linear Regression training completed.")
            return importance

        except Exception as e:
            print(f"Error in training Linear Regression: {str(e)}")
            raise
            
    def train_xgboost(self, X_train, y_train):
        """
        Train an XGBoost model with hyperparameter tuning and return feature importance
        """
        try:
            print("Training XGBoost model...")
        
            # Define base parameters
            base_params = {
             'objective': 'reg:squarederror',
             'random_state': self.random_state,
             'n_jobs': -1  # Use all available cores
            }
        
            # Parameter grid
            param_combinations = [
             {
                'max_depth': max_depth,
                'learning_rate': lr,
                'n_estimators': n_est,
                'subsample': ss,
                'colsample_bytree': cs
             }
             for max_depth in [3, 5, 7]
             for lr in [0.01, 0.1]
             for n_est in [100, 200]
             for ss in [0.8, 1.0]
             for cs in [0.8, 1.0]
            ]
        
            best_score = float('inf')
            best_params = None
            best_model = None
        
            # Manual grid search
            for params in param_combinations:
                try:
                    # Combine base parameters with current parameter combination
                    current_params = {**base_params, **params}
                
                    # Create and train model with current parameters
                    model = xgb.XGBRegressor(**current_params)
                
                    # Simple fit without early stopping
                    model.fit(X_train, y_train)
                
                    # Predict and calculate MSE
                    y_pred = model.predict(X_train)
                    mse = mean_squared_error(y_train, y_pred)
                
                    # Update best model if current one is better
                    if mse < best_score:
                       best_score = mse
                       best_params = params
                       best_model = model
                    
                except Exception as e:
                    print(f"Warning: Failed to train with parameters {params}: {str(e)}")
                    continue
        
            if best_model is None:
                # Create a basic model with default parameters
                print("Warning: Parameter search failed. Training basic model with default parameters...")
                basic_params = {
                   'max_depth': 3,
                   'learning_rate': 0.1,
                   'n_estimators': 100,
                    **base_params
                }
                basic_model = xgb.XGBRegressor(**basic_params)
                basic_model.fit(X_train, y_train)
                self.models['xgboost'] = basic_model
            else:
                print(f"Best parameters found: {best_params}")
                print(f"Best MSE score: {best_score}")
                self.models['xgboost'] = best_model
        
             # Calculate feature importance
            importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.models['xgboost'].feature_importances_
             }).sort_values('importance', ascending=False)

            print("XGBoost training completed successfully.")
            return importance

        except Exception as e:
            print(f"Error in training XGBoost: {str(e)}")
            raise
        
    def cluster_analysis(self, X_train):
        """
        Perform clustering analysis using K-Means and DBSCAN
        """
    
        try:
            print("Performing clustering analysis...")
            clustering_results = {}

            # K-Means clustering
            print("Running K-Means clustering...")
            n_clusters_range = range(2, 11)
            inertias = []

            for n_clusters in n_clusters_range:
                kmeans = KMeans(
                   n_clusters=n_clusters,
                   random_state=self.random_state,
                   n_init=10
                )
                kmeans.fit(X_train)
                inertias.append(kmeans.inertia_)

            # Find optimal number of clusters using elbow method
            optimal_clusters = 5  # Default value
            for i in range(1, len(inertias)-1):
               if (inertias[i-1] - inertias[i]) / (inertias[i] - inertias[i+1]) < 1.5:
                optimal_clusters = i + 2
                break

            # Final K-Means with optimal clusters
            print(f"Running final K-Means with {optimal_clusters} clusters...")
            kmeans_final = KMeans(
                n_clusters=optimal_clusters,
                random_state=self.random_state,
                n_init=10
            )
            kmeans_clusters = kmeans_final.fit_predict(X_train)

            # DBSCAN clustering
            print("Running DBSCAN clustering...")
            # Use nearest neighbors to determine eps parameter
            nn = NearestNeighbors(n_neighbors=2)
            nbrs = nn.fit(X_train)
            distances, indices = nbrs.kneighbors(X_train)
        
            # Calculate eps as the mean of the distances to the nearest neighbor
            eps = np.mean(distances[:, 1]) * 2

            print(f"Calculated eps parameter for DBSCAN: {eps:.4f}")
        
            dbscan = DBSCAN(
               eps=eps,
               min_samples=5,
               n_jobs=-1
            )
            dbscan_clusters = dbscan.fit_predict(X_train)

            # Store results
            clustering_results = {
              'kmeans': {
                'clusters': kmeans_clusters,
                'optimal_n_clusters': optimal_clusters,
                'inertias': inertias,
                'cluster_centers': kmeans_final.cluster_centers_,
                'cluster_sizes': np.bincount(kmeans_clusters)
              },
              'dbscan': {
                'clusters': dbscan_clusters,
                'n_clusters': len(set(dbscan_clusters)) - (1 if -1 in dbscan_clusters else 0),
                'noise_points': sum(dbscan_clusters == -1),
                'eps_used': eps,
                'cluster_sizes': np.bincount(dbscan_clusters[dbscan_clusters >= 0])
              }
            }

            print("Clustering analysis completed successfully.")
            print(f"K-Means found {optimal_clusters} clusters")
            print(f"DBSCAN found {clustering_results['dbscan']['n_clusters']} clusters and {clustering_results['dbscan']['noise_points']} noise points")
        
            return clustering_results

        except Exception as e:
            print(f"Error in clustering analysis: {str(e)}")
            raise
        
    def risk_classification(self, threshold=None):
        # Clean the data first
        cleaned_data = self.clean_data(self.data)
        
        if threshold is None:
            threshold = cleaned_data[self.target_col].median()

        # Create binary risk labels
        risk_labels = (cleaned_data[self.target_col] > threshold).astype(int)

        # Handle missing values
        X = cleaned_data.drop([self.target_col], axis=1).select_dtypes(include=[np.number])
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

        X_train, X_test, y_train, y_test = train_test_split(
            X, risk_labels, test_size=self.test_size, random_state=self.random_state
        )

        # Train logistic regression
        log_reg = LogisticRegression(random_state=self.random_state)
        log_reg.fit(X_train, y_train)

        # Train random forest classifier
        rf_clf = RandomForestClassifier(random_state=self.random_state)
        rf_clf.fit(X_train, y_train)

        # Evaluate classifiers
        results = {}
        for name, model in [('logistic', log_reg), ('random_forest', rf_clf)]:
            y_pred = model.predict(X_test)
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred)
            }

        return results

    def run_full_analysis(self, data):
        """
        Run a comprehensive analysis pipeline including:
        - Data preparation
        - Model training (Linear Regression, XGBoost)
        - Clustering (K-Means, DBSCAN)
        - Risk classification
        - Evaluation of models

        :return: Dictionary containing results of the full analysis
        """
        analysis_results = {}
        
        try:
            # Step 1: Prepare data
            print("Preparing data...")
            X_train, X_test, y_train, y_test, feature_names = self.prepare_data(data)
            
            # Step 2: Train Linear Regression model
            print("Training Linear Regression model...")
            linear_regression_importance = self.train_linear_regression(X_train, y_train)
            analysis_results['linear_regression_importance'] = linear_regression_importance
            
            # Step 3: Train XGBoost model
            print("Training XGBoost model...")
            xgboost_importance = self.train_xgboost(X_train, y_train)
            analysis_results['xgboost_importance'] = xgboost_importance
            
            # Step 4: Perform clustering
            print("Performing clustering analysis...")
            clustering_results = self.cluster_analysis(X_train)
            analysis_results['clustering_results'] = clustering_results
            
            # Step 5: Risk classification
            print("Performing risk classification...")
            risk_classification_results = self.risk_classification()
            analysis_results['risk_classification_results'] = risk_classification_results
            
            # Step 6: Evaluate models
            print("Evaluating models...")
            evaluation_results = {}
            for model_name, model in self.models.items():
                y_pred = model.predict(X_test)
                evaluation_results[model_name] = {
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'r2': r2_score(y_test, y_pred)
                }
            analysis_results['evaluation_results'] = evaluation_results
            
            print("Analysis completed successfully.")
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            raise
            
        return analysis_results
def market_outlook(cleaned_data):
    """
    Analyze Market Outlook and Trends with Enhanced Robustness
    """
    try:
        # Identify time-based or date-like columns
        time_columns = [
            col for col in cleaned_data.columns 
            if ('date' in col.lower() or 
                'time' in col.lower() or 
                'inception' in col.lower() or 
                'performance' in col.lower())
        ]
        
        if not time_columns:
            logging.warning("No time-based columns found for market outlook.")
            return None
        
        # Basic market trend analysis
        market_trends = {}
        for col in time_columns:
            try:
                # Convert column to numeric, handling potential errors
                numeric_col = pd.to_numeric(cleaned_data[col], errors='coerce')
                
                # Skip if all values are NaN
                if numeric_col.isna().all():
                    continue
                
                # Remove NaN values for trend calculation
                valid_values = numeric_col.dropna()
                
                # Ensure we have at least two values
                if len(valid_values) < 2:
                    continue
                
                market_trends[col] = {
                    'start_value': valid_values.iloc[0],
                    'end_value': valid_values.iloc[-1],
                    'trend': 'Positive' if valid_values.iloc[-1] > valid_values.iloc[0] else 'Negative',
                    'change_percentage': ((valid_values.iloc[-1] - valid_values.iloc[0]) / valid_values.iloc[0]) * 100
                }
            except Exception as e:
                logging.warning(f"Could not process column {col}: {e}")
        
        if not market_trends:
            logging.warning("No processable time-based columns found.")
            return None
        
        logging.info("Market outlook analysis completed.")
        return market_trends
    
    except Exception as e:
        logging.error(f"Error in market outlook analysis: {e}")
        return None



class FundRanking:
    def __init__(self, data):
        self.data = data
        self.weights = {
            'returns_3yr': 0.20,
            'returns_5yr': 0.25,
            'risk': 0.15,
            'expense_ratio': 0.15,
            'esg_score': 0.15,
            'sharpe_ratio': 0.10
        }
        
    def normalize_metric(self, series, reverse=False):
        """Normalize metrics to 0-1 scale"""
        if series.empty or series.isna().all():
            return pd.Series(np.nan, index=series.index)
        
        min_val = series.min()
        max_val = series.max()
        
        if min_val == max_val:
            return pd.Series(1.0, index=series.index)
            
        if reverse:
            return (max_val - series) / (max_val - min_val)
        return (series - min_val) / (max_val - min_val)

    def calculate_composite_score(self):
        """Calculate weighted composite score for each fund"""
        scores = pd.DataFrame(index=self.data.index)
        
        # Map column names from the dataset to score components
        column_mapping = {
            'returns_3yr': 'Returns and fees: Month end trailing returns, 3 year',
            'returns_5yr': 'Returns and fees: Month end trailing returns, 5 year',
            'risk': 'return_volatility',  # Assuming this is calculated elsewhere
            'expense_ratio': 'Returns and fees: Prospectus net expense ratio',
            'esg_score': 'composite_esg_score',
            'sharpe_ratio': 'sharpe_ratio'
        }
        
        # Normalize each metric
        for score_name, column_name in column_mapping.items():
            if column_name in self.data.columns:
                scores[score_name] = self.normalize_metric(
                    pd.to_numeric(self.data[column_name], errors='coerce'),
                    reverse=(score_name in ['risk', 'expense_ratio'])
                )
        
        # Calculate weighted score using available metrics
        valid_scores = scores.dropna(how='all', axis=1)
        available_weights = {k: self.weights[k] for k in valid_scores.columns if k in self.weights}
        
        # Normalize weights to sum to 1
        weight_sum = sum(available_weights.values())
        normalized_weights = {k: v/weight_sum for k, v in available_weights.items()}
        
        # Calculate composite score
        composite_score = sum(valid_scores[metric] * weight 
                            for metric, weight in normalized_weights.items())
        
        return composite_score

    def get_top_funds(self, n=10):
        """Get top performing funds based on composite score with detailed metrics"""
        # Calculate composite scores
        composite_scores = self.calculate_composite_score()
        
        # Create DataFrame with fund information and scores
        top_funds = pd.DataFrame({
            'Fund Name': self.data['Fund profile: Fund name'],
            'Ticker': self.data['Fund profile: Ticker'],
            'Asset Manager': self.data['Fund profile: Asset manager'],
            'Composite Score': composite_scores,
            '3-Year Return (%)': pd.to_numeric(self.data['Returns and fees: Month end trailing returns, 3 year'], errors='coerce'),
            '5-Year Return (%)': pd.to_numeric(self.data['Returns and fees: Month end trailing returns, 5 year'], errors='coerce'),
            'Expense Ratio (%)': pd.to_numeric(self.data['Returns and fees: Prospectus net expense ratio'], errors='coerce'),
            'ESG Score': pd.to_numeric(self.data['composite_esg_score'], errors='coerce'),
            'Gender Equality Score': pd.to_numeric(self.data['Gender Equality Funds: Gender equality score - Overall score (out of 100 points)'], errors='coerce'),
            'Carbon Footprint': pd.to_numeric(self.data['Fossil Free Funds: Relative carbon footprint (tonnes CO2 / $1M USD invested)'], errors='coerce')
        })
        
        # Sort by composite score and get top N funds
        top_funds = top_funds.sort_values('Composite Score', ascending=False).head(n)
        
        # Round numeric columns for better presentation
        numeric_columns = ['Composite Score', '3-Year Return (%)', '5-Year Return (%)', 
                         'Expense Ratio (%)', 'ESG Score', 'Gender Equality Score', 
                         'Carbon Footprint']
        
        top_funds[numeric_columns] = top_funds[numeric_columns].round(2)
        
        return top_funds

    def visualize_top_funds(self, top_funds):
        """Create visualizations for top funds analysis"""
        

        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Returns comparison
        top_funds.plot(kind='bar', x='Fund Name', y=['3-Year Return (%)', '5-Year Return (%)'], 
                      ax=axes[0, 0])
        axes[0, 0].set_title('Returns Comparison - Top Funds')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. ESG vs Returns scatter
        sns.scatterplot(data=top_funds, x='ESG Score', y='5-Year Return (%)', 
                       size='Composite Score', ax=axes[0, 1])
        axes[0, 1].set_title('ESG Score vs 5-Year Returns')
        
        # 3. Expense Ratio comparison
        top_funds.plot(kind='bar', x='Fund Name', y='Expense Ratio (%)', 
                      ax=axes[1, 0], color='green')
        axes[1, 0].set_title('Expense Ratio Comparison')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Carbon Footprint vs Gender Equality
        sns.scatterplot(data=top_funds, x='Gender Equality Score', y='Carbon Footprint', 
                       size='Composite Score', ax=axes[1, 1])
        axes[1, 1].set_title('Gender Equality Score vs Carbon Footprint')
        
        plt.tight_layout()
        plt.show()

    def generate_summary_report(self, n=10):
        """Generate a comprehensive summary report for top funds"""
        top_funds = self.get_top_funds(n)
        
        print("=== Top Fund Analysis Report ===\n")
        print(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n")
        print("Top 10 Funds Summary:")
        print("--------------------")
        
        # Print detailed fund information
        for idx, fund in top_funds.iterrows():
            print(f"\n{idx + 1}. {fund['Fund Name']} ({fund['Ticker']})")
            print(f"   Asset Manager: {fund['Asset Manager']}")
            print(f"   Composite Score: {fund['Composite Score']:.2f}")
            print(f"   3-Year Return: {fund['3-Year Return (%)']}%")
            print(f"   5-Year Return: {fund['5-Year Return (%)']}%")
            print(f"   Expense Ratio: {fund['Expense Ratio (%)']}%")
            print(f"   ESG Score: {fund['ESG Score']}")
            print(f"   Gender Equality Score: {fund['Gender Equality Score']}")
            print(f"   Carbon Footprint: {fund['Carbon Footprint']} tonnes CO2/$1M")
        
        # Create visualizations
        self.visualize_top_funds(top_funds)
        
        return top_funds
def recommend_complementary_funds(data, target_fund=None, top_n=5):
    """
    Recommend complementary funds based on correlation analysis of key metrics.
    Returns funds that could provide good diversification benefits.
    
    Parameters:
    - data: DataFrame containing fund data
    - target_fund: Specific fund to analyze (if None, analyzes all top funds)
    - top_n: Number of recommendations to return
    """
    try:
        # Select relevant metrics for correlation analysis
        key_metrics = [
            'Returns and fees: Month end trailing returns, 1 year',
            'Returns and fees: Month end trailing returns, 3 year',
            'Returns and fees: Month end trailing returns, 5 year',
            'Fossil Free Funds: Relative carbon footprint (tonnes CO2 / $1M USD invested)',
            'Gender Equality Funds: Gender equality score - Overall score (out of 100 points)',
            'composite_esg_score'
        ]
        
        # Convert metrics to numeric and handle missing values
        analysis_data = pd.DataFrame()
        for metric in key_metrics:
            if metric in data.columns:
                analysis_data[metric] = pd.to_numeric(data[metric], errors='coerce')
        
        # Drop rows with all missing values
        analysis_data = analysis_data.dropna(how='all')
        
        # Calculate correlation matrix
        correlation_matrix = analysis_data.corr()
        
        # Initialize results dictionary
        recommendations = {
            'complementary_funds': [],
            'correlation_analysis': {},
            'visualization_data': None
        }
        
        # If target fund is specified, analyze correlations for that fund
        if target_fund and target_fund in data.index:
            fund_correlations = correlation_matrix.loc[target_fund]
            
            # Find least correlated funds (best for diversification)
            complementary_funds = fund_correlations.nsmallest(top_n + 1)[1:]  # Exclude self
            
            recommendations['complementary_funds'] = [{
                'fund_name': data.loc[idx, 'Fund profile: Fund name'],
                'ticker': data.loc[idx, 'Fund profile: Ticker'],
                'correlation': corr,
                'key_metrics': {
                    metric: data.loc[idx, metric]
                    for metric in key_metrics
                    if metric in data.columns
                }
            } for idx, corr in complementary_funds.items()]
            
            # Prepare visualization data
            viz_data = pd.DataFrame({
                'Fund': [data.loc[idx, 'Fund profile: Fund name'] for idx in complementary_funds.index],
                'Correlation': complementary_funds.values,
                'ESG Score': [data.loc[idx, 'composite_esg_score'] for idx in complementary_funds.index],
                'Returns 3Y': [data.loc[idx, 'Returns and fees: Month end trailing returns, 3 year'] 
                             for idx in complementary_funds.index]
            })
            recommendations['visualization_data'] = viz_data
            
        else:
            # Analyze all funds
            # Calculate average correlation for each fund
            avg_correlations = correlation_matrix.mean()
            
            # Find funds with lowest average correlations
            diverse_funds = avg_correlations.nsmallest(top_n)
            
            recommendations['complementary_funds'] = [{
                'fund_name': data.loc[idx, 'Fund profile: Fund name'],
                'ticker': data.loc[idx, 'Fund profile: Ticker'],
                'avg_correlation': corr,
                'key_metrics': {
                    metric: data.loc[idx, metric]
                    for metric in key_metrics
                    if metric in data.columns
                }
            } for idx, corr in diverse_funds.items()]
        
        # Add correlation analysis summary
        recommendations['correlation_analysis'] = {
            'avg_correlation': correlation_matrix.mean().mean(),
            'metric_correlations': {
                metric: correlation_matrix[metric].mean()
                for metric in key_metrics
                if metric in correlation_matrix.columns
            }
        }
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='RdYlBu', 
                   center=0,
                   fmt='.2f')
        plt.title('Fund Metrics Correlation Heatmap')
        plt.tight_layout()
        plt.show()
        
        return recommendations
        
    except Exception as e:
        logging.error(f"Error in fund recommendation: {str(e)}")
        return None

def print_complementary_recommendations(recommendations):
    """
    Print formatted recommendations
    """
    if not recommendations:
        print("No recommendations available.")
        return
    
    print("\n=== Fund Complementary Analysis ===")
    print("\nRecommended Complementary Funds:")
    print("---------------------------------")
    
    for i, fund in enumerate(recommendations['complementary_funds'], 1):
        print(f"\n{i}. {fund['fund_name']} ({fund['ticker']})")
        print(f"   Correlation: {fund.get('correlation', fund.get('avg_correlation')):.3f}")
        print("\n   Key Metrics:")
        for metric, value in fund['key_metrics'].items():
            metric_name = metric.split(': ')[-1]  # Get the last part of the metric name
            print(f"   - {metric_name}: {value}")
    
    print("\nCorrelation Analysis Summary:")
    print("---------------------------------")
    print(f"Average Overall Correlation: {recommendations['correlation_analysis']['avg_correlation']:.3f}")
    print("\nMetric-wise Average Correlations:")
    for metric, corr in recommendations['correlation_analysis']['metric_correlations'].items():
        metric_name = metric.split(': ')[-1]
        print(f"- {metric_name}: {corr:.3f}")

# Example usage:
# recommendations = recommend_complementary_funds(data, target_fund='FUND_NAME')
# print_complementary_recommendations(recommendations)
def generate_comprehensive_report(file_path):
    """
    Generate a Comprehensive Investment Analysis Report with Enhanced Logging and Error Handling.
    This function includes multiple analyses such as returns, market outlook, complementary stocks, 
    risk metrics, ESG scores, fund ranking, and models to generate an actionable investment report.
    """
    try:
        # Load and clean data
        raw_data = load_data(file_path)
        
        # Log initial data information
        logging.info(f"Raw Data Shape: {raw_data.shape}")
        logging.info("Raw Data Columns:")
        for col in raw_data.columns:
            logging.info(f"- {col}")
        
        # Clean data
        cleaned_data = clean_data(raw_data)
        
        # Check if cleaned data is None or empty
        if cleaned_data is None or cleaned_data.empty:
            logging.error("No data available after cleaning.")
            return None
        
            
        # Preprocess the data
       

        
        
        # Initialize the report dictionary
        report = {}

        # Returns Analysis
        returns_result = analyze_returns(cleaned_data)
        if returns_result:
            report['returns_analysis'] = returns_result
        else:
            logging.warning("Could not complete returns analysis")
        
        # Market Outlook
        market_outlook_result = market_outlook(cleaned_data)
        if market_outlook_result:
            report['market_outlook'] = market_outlook_result
        else:
            logging.warning("Could not complete market outlook analysis")
        
        # Complementary Stocks
        recommendations = recommend_complementary_funds(cleaned_data)
        print_complementary_recommendations(recommendations)
        
        
        # Perform EDA (if required)
        if cleaned_data is not None:
            enhanced_univariate_analysis(cleaned_data)
           # enhanced_multivariate_analysis(cleaned_data)
            eda_analysis(cleaned_data)

        # Risk metrics and ESG scores calculation
        cleaned_data = calculate_risk_metrics(cleaned_data)
        main_analysis(cleaned_data)
        cleaned_data = create_esg_score(cleaned_data)
        
        # Encode categorical features
        cleaned_data = encode_categorical_features(cleaned_data)

        # Anomalies detection and visualization
        #anomalies = detect_anomalies(cleaned_data)
        #feature_importance = select_features(cleaned_data)
        #visualize_anomalies(cleaned_data, anomalies)

        # Fund Analysis Models
        fund_analysis = FundAnalysisModels(cleaned_data)
        analysis_results = fund_analysis.run_full_analysis(cleaned_data)
        report['model_analysis'] = analysis_results

        # Fund Ranking
        fund_ranking = FundRanking(cleaned_data)
        ranking_results = fund_ranking.generate_summary_report(n = 10)
        report['fund_rankings'] = ranking_results  

        # Check if report is empty
        if not report:
            logging.error("No analyses could be completed.")
            return None
        
        logging.info("Comprehensive report generated successfully.")
        return report
    
    except Exception as e:
        logging.error(f"Error generating comprehensive report: {e}")
        return None
    
def main():
    """
    Main function to execute the comprehensive investment analysis
    """
    try:
        # File path (you might want to make this configurable)
        file_path = r"C:\Users\DELL\Downloads\Dataset.csv"
        
        # Generate comprehensive report
        comprehensive_report = generate_comprehensive_report(file_path)
        
        # Additional processing or printing of report
        if comprehensive_report:
            print("Comprehensive Report Generated Successfully")
            
            # Print out some key insights from the analysis
            
            """if 'returns_analysis' in comprehensive_report:
                print("\nReturns Analysis:")
                for key, value in comprehensive_report['returns_analysis'].items():
                    print(f"{key}: {value}")"""
            
            if 'market_outlook' in comprehensive_report:
                print("\nMarket Outlook:")
                for key, value in comprehensive_report['market_outlook'].items():
                    print(f"{key}: {value}")
            
            
            
            if 'model_analysis' in comprehensive_report:
                print("\nFund Analysis Models:")
                for key, value in comprehensive_report['model_analysis'].items():
                    print(f"{key}: {value}")
            
            if 'fund_rankings' in comprehensive_report:
                print("\nFund Rankings:")
                for key, value in comprehensive_report['fund_rankings'].items():
                    print(f"{key}: {value}")
            
        else:
            logging.error("No report generated.")
            
    except Exception as e:
        logging.error(f"Error in main execution: {e}")

# This ensures the main function is called only when the script is run directly
if __name__ == "__main__":
    main()
