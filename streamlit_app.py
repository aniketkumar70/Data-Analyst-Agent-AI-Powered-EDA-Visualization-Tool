import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import tempfile
from io import StringIO, BytesIO

# File processing libraries
import docx
from PyPDF2 import PdfReader
import openpyxl
from PIL import Image
import requests

# Import the agent class (assuming it's in the same directory)
# from data_analyst_agent import DataAnalystAgent

# Streamlit page configuration
st.set_page_config(
    page_title="Data Analyst Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .insight-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .stAlert > div {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Data Analyst Agent Class (embedded for Streamlit)
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
            response = requests.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"Error querying LLM: {str(e)}"
    
    def load_file_from_uploaded(self, uploaded_file):
        """Load and process uploaded file from Streamlit"""
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        try:
            if file_extension == '.csv':
                self.data = pd.read_csv(uploaded_file)
                self.file_type = 'csv'
                return f"CSV loaded successfully. Shape: {self.data.shape}"
                
            elif file_extension in ['.xlsx', '.xls']:
                self.data = pd.read_excel(uploaded_file)
                self.file_type = 'excel'
                return f"Excel file loaded successfully. Shape: {self.data.shape}"
                
            elif file_extension == '.txt':
                self.file_content = str(uploaded_file.read(), "utf-8")
                self.file_type = 'text'
                return f"Text file loaded successfully. Length: {len(self.file_content)} characters"
                
            elif file_extension == '.docx':
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                
                doc = docx.Document(tmp_file_path)
                self.file_content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                self.file_type = 'docx'
                os.unlink(tmp_file_path)  # Clean up temp file
                return f"Word document loaded successfully. Length: {len(self.file_content)} characters"
                
            elif file_extension == '.pdf':
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                
                reader = PdfReader(tmp_file_path)
                self.file_content = ''
                for page in reader.pages:
                    self.file_content += page.extract_text() + '\n'
                self.file_type = 'pdf'
                os.unlink(tmp_file_path)  # Clean up temp file
                return f"PDF loaded successfully. Length: {len(self.file_content)} characters"
                
            elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                self.file_content = Image.open(uploaded_file)
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
        """Answer questions about the data using LLM"""
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

# Main Streamlit App
def main():
    st.markdown('<h1 class="main-header">🤖 Data Analyst Agent</h1>', unsafe_allow_html=True)
    