import pandas as pd
import numpy as np
import logging
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer




# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s'
)

# Suppress warnings
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

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

# Example usage with specific date columns
# date_columns = ['Date', 'CreatedAt']  # Replace with actual date column names in your dataset
# raw_data = load_data(file_path, date_columns)

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
        
        logging.info("Data cleaned successfully.")
        logging.info(f"Cleaned data shape: {cleaned_data.shape}")
        
        return cleaned_data
    
    except Exception as e:
        logging.error(f"Error in data cleaning: {e}")
        return None


def perform_eda(cleaned_data):
    """
    Comprehensive Exploratory Data Analysis
    """
    try:
        # Correlation heatmap
        plt.figure(figsize=(12, 10))
        correlation_matrix = cleaned_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap of Financial Features')
        plt.tight_layout()
        plt.show()
        
        # Distribution of key numerical features
        numerical_columns = cleaned_data.select_dtypes(include=[np.number]).columns
        cleaned_data[numerical_columns].hist(figsize=(15, 10), bins=20)
        plt.suptitle('Distribution of Numerical Features')
        plt.tight_layout()
        plt.show()
        
        logging.info("Exploratory Data Analysis completed.")
    
    except Exception as e:
        logging.error(f"Error in EDA: {e}")

def analyze_returns(cleaned_data):
    """
    Comprehensive Returns Analysis with Enhanced Column Detection
    """
    try:
        # Expanded list of potential return column names
        return_columns = [
            'Returns', 'Return', 'Total Return', 
            'Annual Return', 'Yearly Return', 
            'Net Return', 'ROI', 'Return Rate',
            'Month end trailing returns, year-to-date'
        ]
        
        # Find columns that might represent returns
        potential_return_cols = []
        for col in cleaned_data.columns:
            # Check for columns that might represent returns
            if any(keyword.lower() in col.lower() for keyword in return_columns):
                # Additional check to ensure column contains numeric data
                try:
                    numeric_data = pd.to_numeric(cleaned_data[col], errors='coerce')
                    if not numeric_data.isna().all():
                        potential_return_cols.append(col)
                except:
                    continue
        
        # If no columns found, try to find numeric columns that look like returns
        if not potential_return_cols:
            # Look for numeric columns with percentage-like values
            potential_return_cols = [
                col for col in cleaned_data.select_dtypes(include=[np.number]).columns
                if (cleaned_data[col].between(-100, 100).any() and  # Some values in percentage range
                    not cleaned_data[col].isna().all())  # Not all NaN
            ]
        
        # If still no columns found, log detailed column information
        if not potential_return_cols:
            logging.warning("No return columns detected. Available columns:")
            for col in cleaned_data.columns:
                logging.warning(f"Column: {col}, Type: {cleaned_data[col].dtype}, Sample values: {cleaned_data[col].head()}")
            return None
        
        # Initialize results for multiple return columns
        returns_analysis = {}
        
        # Analyze each potential return column
        for return_col in potential_return_cols:
            # Convert to numeric, coercing errors
            numeric_returns = pd.to_numeric(cleaned_data[return_col], errors='coerce')
            
            # Skip if all values are NaN
            if numeric_returns.isna().all():
                continue
            
            # Calculate return statistics
            col_analysis = {
                'column_name': return_col,
                'mean_return': numeric_returns.mean(),
                'median_return': numeric_returns.median(),
                'return_volatility': numeric_returns.std(),
                'min_return': numeric_returns.min(),
                'max_return': numeric_returns.max(),
                'positive_return_percentage': (numeric_returns > 0).mean() * 100,
                'non_nan_count': numeric_returns.count()
            }
            
            returns_analysis[return_col] = col_analysis
            
            # Visualization for each valid return column
            plt.figure(figsize=(10, 6))
            numeric_returns.hist(bins=30)
            plt.title(f'Distribution of {return_col}')
            plt.xlabel('Returns')
            plt.ylabel('Frequency')
            plt.show()
        
        # If no valid return columns found
        if not returns_analysis:
            logging.warning("No valid return columns found after processing.")
            return None
        
        logging.info(f"Returns analysis completed for {len(returns_analysis)} column(s)")
        return returns_analysis
    
    except Exception as e:
        logging.error(f"Error in returns analysis: {e}")
        return None
        
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
        
def recommend_complementary_stocks(cleaned_data, top_n=5):
    """
    Recommend Complementary Stocks Based on Correlation
    """
    try:
        # Select numerical columns for correlation
        numerical_columns = cleaned_data.select_dtypes(include=[np.number]).columns
        
        # Calculate correlation matrix
        correlation_matrix = cleaned_data[numerical_columns].corr()
        
        # Find top complementary stocks
        complementary_recommendations = {}
        for col in numerical_columns:
            # Find most positively and negatively correlated stocks
            positive_corr = correlation_matrix[col].nlargest(top_n)
            negative_corr = correlation_matrix[col].nsmallest(top_n)
            
            complementary_recommendations[col] = {
                'positive_corr': positive_corr,
                'negative_corr': negative_corr
            }
        
        logging.info("Complementary stock recommendations generated.")
        return complementary_recommendations
    
    except Exception as e:
        logging.error(f"Error in stock recommendation: {e}")
        return None

def generate_comprehensive_report(file_path):
    """
    Generate a Comprehensive Investment Analysis Report with Enhanced Logging and Error Handling
    """
    try:
        # Load and clean data
        raw_data = load_data(file_path)
        
        # Log initial data information
        logging.info(f"Raw Data Shape: {raw_data.shape}")
        logging.info("Raw Data Columns:")
        for col in raw_data.columns:
            logging.info(f"- {col}")
        
        cleaned_data = clean_data(raw_data)
        
        # Check if cleaned data is None or empty
        if cleaned_data is None or cleaned_data.empty:
            logging.error("No data available after cleaning.")
            return None
        
        # Perform analyses with more robust error handling
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
        complementary_stocks_result = recommend_complementary_stocks(cleaned_data)
        if complementary_stocks_result:
            report['complementary_stocks'] = complementary_stocks_result
        else:
            logging.warning("Could not complete complementary stocks analysis")
        
        # Perform EDA
        perform_eda(cleaned_data)
        
        # Check if report is empty
        if not report:
            logging.error("No analyses could be completed.")
            return None
        
        logging.info("Comprehensive report generated successfully.")
        return report
    
    except Exception as e:
        logging.error(f"Error generating comprehensive report: {e}")
        return None


# ... (previous imports and function definitions remain the same)

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
            
            # Optionally print out some key insights
            if comprehensive_report['returns_analysis']:
                print("\nReturns Analysis:")
                for key, value in comprehensive_report['returns_analysis'].items():
                    print(f"{key}: {value}")
            
            if comprehensive_report['market_outlook']:
                print("\nMarket Outlook:")
                for key, value in comprehensive_report['market_outlook'].items():
                    print(f"{key}: {value}")
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}")

# This ensures the main function is called only when the script is run directly
if __name__ == "__main__":
    main()
