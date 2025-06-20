# Data Analyst Agent
# Comprehensive agent for data analysis, visualization, and Q&A across multiple file types
# Uses meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 from Together.ai
#%pip install pandas numpy matplotlib seaborn plotly python-docx PyPDF2 openpyxl Pillow scikit-learn scipy requests

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# File processing libraries

import docx
from PyPDF2 import PdfReader
import openpyxl
from PIL import Image
import base64
from io import BytesIO

# ML and statistics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
import requests

# Together.ai API setup
TOGETHER_API_KEY = "79314aaef9e513592f21b34d607dc0914984797e327bc8498f2ff8dfdfc4bb86"  # Replace with your actual API key
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

class DataAnalystAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        self.data = None
        self.file_type = None
        self.file_content = None
        self.analysis_history = []
        self.conversation_context = []
        
    def query_llama(self, prompt, max_tokens=2000):
        """Query the Llama model via Together.ai API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            "messages": [
                {
                    "role": "system",
                    "content": """You are an expert data analyst AI assistant. You help users analyze data, create insights, 
                    and answer questions about their datasets. You should be precise, insightful, and provide actionable recommendations.
                    When discussing data analysis, focus on:
                    1. Key patterns and trends
                    2. Statistical significance
                    3. Business implications
                    4. Recommendations for further analysis
                    5. Data quality observations
                    Keep responses concise but comprehensive."""
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "top_p": 0.9
        }
        
        try:
            response = requests.post(TOGETHER_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"Error querying LLM: {str(e)}"
    
    def load_file(self, file_path):
        """Load and process different file types"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.csv':
                self.data = pd.read_csv(file_path)
                self.file_type = 'csv'
                return f"CSV loaded successfully. Shape: {self.data.shape}"
                
            elif file_extension == '.xlsx' or file_extension == '.xls':
                self.data = pd.read_excel(file_path)
                self.file_type = 'excel'
                return f"Excel file loaded successfully. Shape: {self.data.shape}"
                
            elif file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.file_content = f.read()
                self.file_type = 'text'
                return f"Text file loaded successfully. Length: {len(self.file_content)} characters"
                
            elif file_extension == '.docx':
                doc = docx.Document(file_path)
                self.file_content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                self.file_type = 'docx'
                return f"Word document loaded successfully. Length: {len(self.file_content)} characters"
                
            elif file_extension == '.pdf':
                reader = PdfReader(file_path)
                self.file_content = ''
                for page in reader.pages:
                    self.file_content += page.extract_text() + '\n'
                self.file_type = 'pdf'
                return f"PDF loaded successfully. Length: {len(self.file_content)} characters"
                
            elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                self.file_content = Image.open(file_path)
                self.file_type = 'image'
                return f"Image loaded successfully. Size: {self.file_content.size}"
                
            else:
                return f"Unsupported file type: {file_extension}"
                
        except Exception as e:
            return f"Error loading file: {str(e)}"
    
    def get_data_summary(self):
        """Generate comprehensive data summary"""
        if self.data is None:
            return "No structured data loaded."
        
        summary = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'numeric_summary': self.data.describe().to_dict() if not self.data.select_dtypes(include=[np.number]).empty else {},
            'categorical_summary': {}
        }
        
        # Categorical data summary
        cat_cols = self.data.select_dtypes(include=['object']).columns
        for col in cat_cols:
            summary['categorical_summary'][col] = {
                'unique_count': self.data[col].nunique(),
                'top_values': self.data[col].value_counts().head().to_dict()
            }
        
        return summary
    
    def auto_visualize(self):
        """Automatically create relevant visualizations based on data"""
        if self.data is None:
            return "No data to visualize"
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Automated Data Analysis Dashboard', fontsize=16)
        
        # 1. Missing values heatmap
        if self.data.isnull().sum().sum() > 0:
            sns.heatmap(self.data.isnull(), ax=axes[0,0], cbar=True, yticklabels=False)
            axes[0,0].set_title('Missing Values Pattern')
        else:
            axes[0,0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=12)
            axes[0,0].set_title('Missing Values Pattern')
        
        # 2. Correlation heatmap for numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = self.data[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0,1])
            axes[0,1].set_title('Correlation Matrix')
        else:
            axes[0,1].text(0.5, 0.5, 'Insufficient Numeric Data', ha='center', va='center', fontsize=12)
            axes[0,1].set_title('Correlation Matrix')
        
        # 3. Distribution of first numeric column
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            self.data[col].hist(bins=30, ax=axes[1,0], alpha=0.7)
            axes[1,0].set_title(f'Distribution of {col}')
            axes[1,0].set_xlabel(col)
            axes[1,0].set_ylabel('Frequency')
        else:
            axes[1,0].text(0.5, 0.5, 'No Numeric Data', ha='center', va='center', fontsize=12)
            axes[1,0].set_title('Data Distribution')
        
        # 4. Top categories for first categorical column
        cat_cols = self.data.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            col = cat_cols[0]
            top_cats = self.data[col].value_counts().head(10)
            top_cats.plot(kind='bar', ax=axes[1,1])
            axes[1,1].set_title(f'Top 10 {col}')
            axes[1,1].tick_params(axis='x', rotation=45)
        else:
            axes[1,1].text(0.5, 0.5, 'No Categorical Data', ha='center', va='center', fontsize=12)
            axes[1,1].set_title('Category Distribution')
        
        plt.tight_layout()
        plt.show()
        
        return "Automated visualizations generated successfully"
    
    def create_custom_visualization(self, viz_type, columns=None, title="Custom Visualization"):
        """Create specific visualizations based on user request"""
        if self.data is None:
            return "No data available for visualization"
        
        try:
            if viz_type.lower() == 'scatter':
                if columns and len(columns) >= 2:
                    fig = px.scatter(self.data, x=columns[0], y=columns[1], title=title)
                    fig.show()
                    return f"Scatter plot created for {columns[0]} vs {columns[1]}"
                
            elif viz_type.lower() == 'line':
                if columns:
                    fig = px.line(self.data, y=columns, title=title)
                    fig.show()
                    return f"Line plot created for {columns}"
                
            elif viz_type.lower() == 'bar':
                if columns:
                    col = columns[0]
                    value_counts = self.data[col].value_counts().head(20)
                    fig = px.bar(x=value_counts.index, y=value_counts.values, title=title)
                    fig.show()
                    return f"Bar chart created for {col}"
                
            elif viz_type.lower() == 'histogram':
                if columns:
                    fig = px.histogram(self.data, x=columns[0], title=title)
                    fig.show()
                    return f"Histogram created for {columns[0]}"
                
            elif viz_type.lower() == 'box':
                if columns:
                    fig = px.box(self.data, y=columns[0], title=title)
                    fig.show()
                    return f"Box plot created for {columns[0]}"
                    
            else:
                return f"Visualization type '{viz_type}' not supported. Try: scatter, line, bar, histogram, box"
                
        except Exception as e:
            return f"Error creating visualization: {str(e)}"
    
    def perform_statistical_analysis(self):
        """Perform statistical analysis on the data"""
        if self.data is None:
            return "No data available for analysis"
        
        analysis_results = {}
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        # Basic statistics
        if len(numeric_cols) > 0:
            analysis_results['descriptive_stats'] = self.data[numeric_cols].describe().to_dict()
            
            # Correlation analysis
            if len(numeric_cols) > 1:
                corr_matrix = self.data[numeric_cols].corr()
                # Find highly correlated pairs
                high_corr_pairs = []
                for i, col1 in enumerate(corr_matrix.columns):
                    for j, col2 in enumerate(corr_matrix.columns):
                        if i < j and abs(corr_matrix.iloc[i, j]) > 0.7:
                            high_corr_pairs.append((col1, col2, corr_matrix.iloc[i, j]))
                
                analysis_results['high_correlations'] = high_corr_pairs
            
            # Outlier detection using IQR method
            outliers_info = {}
            for col in numeric_cols:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
                outliers_info[col] = len(outliers)
            
            analysis_results['outliers_count'] = outliers_info
        
        # Categorical analysis
        cat_cols = self.data.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            cat_analysis = {}
            for col in cat_cols:
                cat_analysis[col] = {
                    'unique_values': self.data[col].nunique(),
                    'most_frequent': self.data[col].mode().iloc[0] if not self.data[col].mode().empty else None,
                    'frequency_distribution': self.data[col].value_counts().head().to_dict()
                }
            analysis_results['categorical_analysis'] = cat_analysis
        
        return analysis_results
    
    def answer_question(self, question):
      #Answer questions about the data using LLM
        # Prepare context about the data
        context = ""
        
        if self.data is not None:
            summary = self.get_data_summary()
            stats = self.perform_statistical_analysis()
            
            context = f"""
            Data Summary:
            - Shape: {summary['shape']}
            - Columns: {summary['columns']}
            - Data types: {summary['dtypes']}
            - Missing values: {summary['missing_values']}
            
            Statistical Analysis:
            {json.dumps(stats, indent=2, default=str)}
            
            Sample of the data:
            {self.data.head().to_string()}
            """
        elif self.file_content is not None:
            context = f"""
            File type: {self.file_type}
            Content preview (first 1000 characters):
            {self.file_content[:1000]}
            """
        
        # Add conversation history for context
        conversation_context = "\n".join(self.conversation_context[-5:])  # Last 5 exchanges
        
        prompt = f"""
        Based on the following data context, please answer the user's question:
        
        {context}
        
        Previous conversation:
        {conversation_context}
        
        User Question: {question}
        
        Please provide a comprehensive answer that includes:
        1. Direct answer to the question
        2. Supporting evidence from the data
        3. Any relevant insights or patterns
        4. Recommendations if applicable
        """
        
        response = self.query_llama(prompt)
        
        # Store in conversation history
        self.conversation_context.append(f"Q: {question}")
        self.conversation_context.append(f"A: {response}")
        
        return response
    
    def generate_insights(self):
        """Generate automated insights about the data"""
        if self.data is None:
            return "No structured data available for insights"
        
        summary = self.get_data_summary()
        stats = self.perform_statistical_analysis()
        
        prompt = f"""
        Analyze the following dataset and provide key insights:
        
        Dataset Summary:
        {json.dumps(summary, indent=2, default=str)}
        
        Statistical Analysis:
        {json.dumps(stats, indent=2, default=str)}
        
        Please provide:
        1. 3-5 key insights about the data
        2. Data quality observations
        3. Interesting patterns or anomalies
        4. Recommendations for further analysis
        5. Business implications (if applicable)
        
        Be specific and reference actual values from the data.
        """
        
        insights = self.query_llama(prompt, max_tokens=1500)
        self.analysis_history.append(insights)
        
        return insights

# Initialize the agent
print("=== Data Analyst Agent ===")
print("Intelligent agent for comprehensive data analysis")
print("\nTo get started:")
print("1. Set your Together.ai API key")
print("2. Load a file using agent.load_file('path/to/file')")
print("3. Start analyzing with agent.auto_visualize() or agent.answer_question()")
print("\nSupported file types: CSV, Excel, TXT, DOCX, PDF, Images")

# Create agent instance
agent = DataAnalystAgent(TOGETHER_API_KEY)

# Example usage (uncomment to use):
# Replace with your actual API key
agent = DataAnalystAgent("79314aaef9e513592f21b34d607dc0914984797e327bc8498f2ff8dfdfc4bb86")

# Load a file
result = agent.load_file("netflix_titles.csv")
print(result)

# Get automatic insights
insights = agent.generate_insights()
print(insights)

# Create visualizations
agent.auto_visualize()

# Ask questions
answer = agent.answer_question("What are the main trends in this data?")
print(answer)

# Create custom visualizations
agent.create_custom_visualization('scatter', ['column1', 'column2'])
