import os
import json
import re
import sys
import io
import contextlib
import warnings
from typing import Optional, List, Any, Tuple, Dict
from PIL import Image
import streamlit as st
import pandas as pd
import base64
from io import BytesIO
from together import Together
from e2b_code_interpreter import Sandbox
import datetime
import requests
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)

# Sample datasets configuration
SAMPLE_DATASETS = {
    "Student Mental Health Survey": {
        "path": "sample_datasets/Student Mental health.csv",
        "description": "📚 Student mental health data with depression, anxiety, and academic performance metrics",
        "preview_query": "What patterns exist in student mental health across different courses and academic years?",
        "human_context_hint": "🎓 Academic pressure varies by field - Engineering students face intense technical demands, while Psychology students study human behavior but may internalize stress. Consider how cultural background (Islamic education vs secular courses) affects mental health stigma and help-seeking behavior. Age and marital status create different life pressures - younger students face identity formation while older students juggle family responsibilities.",
        "key_insights": [
            "🧠 Engineering students show higher depression rates due to intense academic pressure",
            "💍 Married students face unique stressors balancing family and studies", 
            "🎯 Psychology students ironically report high anxiety despite studying mental health",
            "📊 CGPA ranges reveal academic stress patterns - lower grades correlate with mental health issues",
            "🌍 Cultural factors in Islamic education may affect help-seeking behavior"
        ]
    },
    "Google Play Store Apps": {
        "path": "sample_datasets/googleplaystore.csv", 
        "description": "📱 Google Play Store app data with ratings, reviews, categories, and download statistics",
        "preview_query": "What factors influence app success and user satisfaction in the Google Play Store?",
        "human_context_hint": "📱 Human psychology drives app success - Family apps dominate because parents prioritize children's education and entertainment. Free apps get more downloads but paid apps often have higher ratings (quality vs quantity). User behavior shows emotional attachment - people rate apps they use daily higher. Content ratings reflect societal values and parental concerns. App categories reveal human needs: productivity apps for work stress, lifestyle apps for self-improvement, and communication apps for social connection.",
        "key_insights": [
            "👨‍👩‍👧‍👦 Family apps dominate (1972 apps) - parents prioritize children's digital needs",
            "💰 Free apps get more downloads but paid apps often have higher ratings",
            "⭐ Users emotionally invest in daily-use apps, leading to higher ratings",
            "🎮 Gaming apps are popular but face fierce competition and rating pressure",
            "📊 Medical apps show trust issues - lower ratings despite critical importance"
        ]
    },
    "Adidas US Sales Data": {
        "path": "sample_datasets/Adidas US Sales Datasets.xlsx",
        "description": "👟 Adidas US sales data with product categories, regions, and sales performance",
        "preview_query": "What are the sales patterns and performance trends for Adidas products across different regions?",
        "human_context_hint": "👟 Fashion and lifestyle choices drive sales - people buy athletic wear for status, not just sports. Regional preferences reflect cultural identity and climate - Northeast urban areas prefer streetwear for fashion, while West Coast focuses on athletic performance. Seasonal patterns show human behavior: New Year fitness resolutions, spring outdoor activities, and holiday gift-giving. Price sensitivity varies by region - affluent areas pay premium for brand status. Sales methods reflect shopping psychology - in-store experiences vs online convenience.",
        "key_insights": [
            "🏙️ Northeast urban areas prefer streetwear for fashion status over athletic performance",
            "🌤️ Seasonal patterns reflect human behavior: New Year resolutions, spring activities",
            "💰 Price sensitivity varies by region - affluent areas pay premium for brand status",
            "🛍️ In-store vs online sales reflect shopping psychology and experience preferences",
            "🎯 Product categories show lifestyle choices, not just athletic needs"
        ]
    }
}

class HumanIntelligenceLayer:
    """The core Human Intelligence layer that adds context, emotion, and intuition to AI analysis"""
    
    def __init__(self):
        self.cultural_contexts = {
            "holiday_seasons": ["December", "January", "November", "July"],
            "work_patterns": ["Monday", "Friday", "weekend"],
            "life_events": ["graduation", "wedding", "birth", "death", "divorce", "job_change"]
        }
        
    def analyze_emotional_context(self, data_description: str) -> Dict[str, Any]:
        """Analyze the emotional undertones of data patterns"""
        blob = TextBlob(data_description)
        sentiment = blob.sentiment
        
        emotional_insights = {
            "sentiment_score": sentiment.polarity,
            "emotional_tone": "positive" if sentiment.polarity > 0.1 else "negative" if sentiment.polarity < -0.1 else "neutral",
            "subjectivity": sentiment.subjectivity,
            "human_interpretation": self._get_human_interpretation(sentiment.polarity)
        }
        
        return emotional_insights
    
    def _get_human_interpretation(self, polarity: float) -> str:
        """Convert sentiment scores to human-readable interpretations"""
        if polarity > 0.5:
            return "This data suggests excitement, growth, or positive human experiences"
        elif polarity > 0.1:
            return "Mildly positive patterns - people seem content or optimistic"
        elif polarity < -0.5:
            return "Strong negative patterns - stress, dissatisfaction, or challenges"
        elif polarity < -0.1:
            return "Some concerning trends - worth investigating human factors"
        else:
            return "Neutral patterns - routine human behavior, no strong emotions detected"
    
    def suggest_human_context(self, df: pd.DataFrame, query: str) -> List[str]:
        """Suggest possible human contexts that AI might miss"""
        suggestions = []
        
        # Date-based context
        if any(col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()):
            suggestions.append("🗓️ Consider seasonal patterns, holidays, or major world events during this time period")
            suggestions.append("📅 Weekend vs weekday human behavior patterns might explain anomalies")
        
        # Financial data context
        if any(col for col in df.columns if any(word in col.lower() for word in ['price', 'cost', 'revenue', 'sales', 'money', 'budget'])):
            suggestions.append("💰 Economic stress, paycheck cycles, or cultural spending habits could influence this data")
            suggestions.append("🛍️ Consumer psychology and emotional spending patterns might be at play")
        
        # Social media or engagement context
        if any(col for col in df.columns if any(word in col.lower() for word in ['likes', 'shares', 'comments', 'views', 'engagement'])):
            suggestions.append("📱 Viral trends, social media algorithms, or cultural moments could skew this data")
            suggestions.append("👥 Human attention spans and social validation behaviors are complex")
        
        # Performance or productivity context
        if any(col for col in df.columns if any(word in col.lower() for word in ['performance', 'productivity', 'efficiency', 'score', 'rating'])):
            suggestions.append("🧠 Mental health, work-life balance, or team dynamics could explain these patterns")
            suggestions.append("💪 Human motivation cycles and burnout patterns are often invisible to AI")
        
        # Mental health specific context
        if any(col for col in df.columns if any(word in col.lower() for word in ['depression', 'anxiety', 'mental', 'health'])):
            suggestions.append("🧠 Cultural stigma around mental health affects reporting accuracy and help-seeking behavior")
            suggestions.append("🎓 Academic pressure varies dramatically by field - engineering vs arts vs business")
            suggestions.append("💍 Life stage matters - married students face different stressors than single students")
            suggestions.append("🌍 Cultural and religious background significantly influences mental health perceptions")
            suggestions.append("📚 GPA stress creates a vicious cycle - poor mental health leads to lower grades")
        
        # App store specific context
        if any(col for col in df.columns if any(word in col.lower() for word in ['app', 'rating', 'review', 'install', 'category'])):
            suggestions.append("📱 Human psychology drives app success - emotional attachment vs utility")
            suggestions.append("👨‍👩‍👧‍👦 Family apps dominate because parents prioritize children's digital needs")
            suggestions.append("💰 Free vs paid apps reflect different user psychology and expectations")
            suggestions.append("⭐ Rating inflation - users rate apps they use daily higher due to emotional investment")
            suggestions.append("🎮 Gaming apps face fierce competition and rating pressure from passionate users")
        
        # Sales/retail specific context
        if any(col for col in df.columns if any(word in col.lower() for word in ['sales', 'retailer', 'product', 'region', 'price'])):
            suggestions.append("👟 Fashion and lifestyle choices drive sales more than athletic needs")
            suggestions.append("🏙️ Regional preferences reflect cultural identity, climate, and urban vs rural lifestyles")
            suggestions.append("🌤️ Seasonal patterns show human behavior: resolutions, holidays, weather changes")
            suggestions.append("💰 Price sensitivity varies by region - affluent areas pay premium for status")
            suggestions.append("🛍️ Shopping methods reflect psychology - in-store experience vs online convenience")
        
        return suggestions
    
    def generate_bias_alerts(self, ai_analysis: str) -> List[str]:
        """Generate alerts about potential AI biases"""
        alerts = []
        
        if "correlation" in ai_analysis.lower():
            alerts.append("⚠️ AI found correlations, but correlation ≠ causation. What human factors could be the real cause?")
        
        if "trend" in ai_analysis.lower() or "pattern" in ai_analysis.lower():
            alerts.append("🎭 AI sees patterns, but humans create them. What emotions or cultural shifts drove this trend?")
        
        if "average" in ai_analysis.lower() or "mean" in ai_analysis.lower():
            alerts.append("📊 Averages hide individual human stories. Who are the outliers and why are they different?")
        
        if "increase" in ai_analysis.lower() or "decrease" in ai_analysis.lower():
            alerts.append("📈 Changes in data often reflect changes in human behavior, not just business metrics")
        
        return alerts

def get_human_intuition_prompt(df: pd.DataFrame, query: str) -> str:
    """Generate prompts to capture human intuition before showing AI analysis"""
    sample_data = df.head(3).to_string()
    
    return f"""
    🧠 **HUMAN INTUITION CHECK**
    
    Before we let AI analyze this data, what does YOUR human brain think?
    
    **Quick Data Preview:**
    ```
    {sample_data}
    ```
    
    **Your Question:** {query}
    
    **What's your gut feeling?**
    """

def extract_technical_patterns(ai_response: str) -> List[str]:
    """Extract technical patterns and statistical findings from AI response"""
    patterns = []
    
    # Look for statistical patterns
    if "correlation" in ai_response.lower():
        patterns.append("📊 **Correlation Analysis:** AI identified relationships between variables")
    
    if "trend" in ai_response.lower():
        patterns.append("📈 **Trend Analysis:** AI detected patterns over time or sequences")
    
    if "average" in ai_response.lower() or "mean" in ai_response.lower():
        patterns.append("📊 **Central Tendency:** AI calculated averages and central values")
    
    if "distribution" in ai_response.lower():
        patterns.append("📊 **Distribution Analysis:** AI examined data spread and patterns")
    
    if "outlier" in ai_response.lower():
        patterns.append("🔍 **Outlier Detection:** AI identified unusual data points")
    
    if "cluster" in ai_response.lower() or "group" in ai_response.lower():
        patterns.append("🎯 **Clustering:** AI grouped similar data points")
    
    if "increase" in ai_response.lower() or "decrease" in ai_response.lower():
        patterns.append("📈 **Change Detection:** AI identified growth or decline patterns")
    
    if "percentage" in ai_response.lower() or "%" in ai_response.lower():
        patterns.append("📊 **Percentage Analysis:** AI calculated proportional changes")
    
    if "regression" in ai_response.lower():
        patterns.append("📈 **Regression Analysis:** AI modeled relationships between variables")
    
    if "variance" in ai_response.lower() or "standard deviation" in ai_response.lower():
        patterns.append("📊 **Variability Analysis:** AI measured data spread and consistency")
    
    if "median" in ai_response.lower():
        patterns.append("📊 **Median Analysis:** AI identified middle values and central tendency")
    
    if "mode" in ai_response.lower():
        patterns.append("📊 **Mode Analysis:** AI found most frequent values")
    
    if "range" in ai_response.lower():
        patterns.append("📊 **Range Analysis:** AI calculated data spread from min to max")
    
    if "quartile" in ai_response.lower():
        patterns.append("📊 **Quartile Analysis:** AI examined data distribution in quarters")
    
    if "skew" in ai_response.lower():
        patterns.append("📊 **Skewness Analysis:** AI measured data distribution asymmetry")
    
    if "kurtosis" in ai_response.lower():
        patterns.append("📊 **Kurtosis Analysis:** AI measured data distribution peakedness")
    
    if "anova" in ai_response.lower() or "f-test" in ai_response.lower():
        patterns.append("📊 **ANOVA Analysis:** AI compared means across multiple groups")
    
    if "chi-square" in ai_response.lower() or "chi-squared" in ai_response.lower():
        patterns.append("📊 **Chi-Square Test:** AI tested categorical variable relationships")
    
    if "t-test" in ai_response.lower():
        patterns.append("📊 **T-Test Analysis:** AI compared means between groups")
    
    if "p-value" in ai_response.lower() or "significance" in ai_response.lower():
        patterns.append("📊 **Statistical Significance:** AI tested hypothesis validity")
    
    if "confidence interval" in ai_response.lower():
        patterns.append("📊 **Confidence Intervals:** AI estimated parameter ranges")
    
    if "r-squared" in ai_response.lower() or "r²" in ai_response.lower():
        patterns.append("📊 **R-Squared Analysis:** AI measured model fit quality")
    
    # Look for specific numerical findings
    import re
    numbers = re.findall(r'\d+\.?\d*%?', ai_response)
    if numbers:
        patterns.append(f"🔢 **Numerical Findings:** AI identified key metrics: {', '.join(numbers[:3])}")
    
    # Look for visualization patterns
    if "histogram" in ai_response.lower():
        patterns.append("📊 **Histogram Analysis:** AI examined data distribution visually")
    
    if "scatter plot" in ai_response.lower() or "scatterplot" in ai_response.lower():
        patterns.append("📊 **Scatter Plot Analysis:** AI visualized variable relationships")
    
    if "box plot" in ai_response.lower() or "boxplot" in ai_response.lower():
        patterns.append("📊 **Box Plot Analysis:** AI examined data distribution and outliers")
    
    if "bar chart" in ai_response.lower() or "bar graph" in ai_response.lower():
        patterns.append("📊 **Bar Chart Analysis:** AI compared categorical data")
    
    if "line chart" in ai_response.lower() or "line graph" in ai_response.lower():
        patterns.append("📊 **Line Chart Analysis:** AI tracked trends over time")
    
    return patterns

def extract_human_insights(human_intuition: str, human_suggestions: List[str]) -> List[str]:
    """Extract unique human insights from intuition and suggestions"""
    insights = []
    
    # Extract insights from human intuition
    if human_intuition:
        # Look for emotional or contextual keywords
        emotional_keywords = ['feel', 'think', 'believe', 'seem', 'probably', 'might', 'could', 'should', 'would', 'may']
        contextual_keywords = ['because', 'since', 'due to', 'as a result', 'therefore', 'however', 'although', 'while', 'when']
        behavioral_keywords = ['people', 'users', 'customers', 'students', 'behavior', 'tend', 'prefer', 'choose', 'decide']
        
        sentences = human_intuition.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword in sentence.lower() for keyword in emotional_keywords + contextual_keywords + behavioral_keywords):
                if len(sentence) > 15:  # Only include substantial insights
                    insights.append(f"💭 **Intuition:** {sentence}")
    
    # Extract insights from human suggestions
    for suggestion in human_suggestions:
        # Prioritize key insights and contextual suggestions
        if "🎯" in suggestion or "🔍" in suggestion:
            insights.append(suggestion)
        elif any(keyword in suggestion.lower() for keyword in [
            'human', 'behavior', 'psychology', 'culture', 'emotion', 'social', 
            'lifestyle', 'preference', 'motivation', 'stress', 'pressure', 'trend',
            'seasonal', 'regional', 'demographic', 'generational', 'economic'
        ]):
            insights.append(suggestion)
    
    # Add dataset-specific insights based on content
    if any(word in human_intuition.lower() for word in ['mental', 'health', 'depression', 'anxiety']):
        insights.append("🧠 **Mental Health Context:** Cultural stigma and academic pressure significantly impact reporting accuracy")
    
    if any(word in human_intuition.lower() for word in ['app', 'rating', 'download', 'store']):
        insights.append("📱 **App Psychology:** User behavior driven by emotional attachment and social validation")
    
    if any(word in human_intuition.lower() for word in ['sales', 'purchase', 'buy', 'retail']):
        insights.append("🛍️ **Consumer Behavior:** Fashion choices reflect lifestyle and status, not just functional needs")
    
    return insights[:6]  # Limit to top 6 insights

def create_human_vs_ai_comparison(human_input: str, ai_analysis: str, emotional_context: Dict) -> str:
    """Create a comparison between human intuition and AI analysis"""
    
    comparison = f"""
    ## 🤖 vs 🧠 **AI vs Human Intelligence Analysis**
    
    ### 🧠 **Your Human Intuition:**
    {human_input}
    
    ### 🤖 **AI's Technical Analysis:**
    {ai_analysis}
    
    ### 💭 **Emotional Context Layer:**
    - **Emotional Tone:** {emotional_context.get('emotional_tone', 'unknown')}
    - **Human Interpretation:** {emotional_context.get('human_interpretation', 'No clear emotional pattern')}
    - **Subjectivity Level:** {'High (very human/personal)' if emotional_context.get('subjectivity', 0) > 0.5 else 'Low (more objective)'}
    
    ### 🎯 **The Human Edge:**
    Where human intelligence might outshine AI in this analysis...
    """
    
    return comparison

def create_story_first_visualization(df: pd.DataFrame, title: str, emotional_tone: str) -> go.Figure:
    """Create visualizations that prioritize emotional impact over technical precision"""
    
    # Choose colors based on emotional tone
    if emotional_tone == "positive":
        colors = px.colors.sequential.Viridis
        bg_color = "#f0f9ff"
    elif emotional_tone == "negative":
        colors = px.colors.sequential.Reds
        bg_color = "#fef2f2"
    else:
        colors = px.colors.sequential.Greys
        bg_color = "#f9fafb"
    
    # Create a figure with story-telling elements
    fig = go.Figure()
    
    # Add some "human imperfection" to the styling
    fig.update_layout(
        title={
            'text': f"📖 {title}",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'family': 'Arial, sans-serif'}
        },
        plot_bgcolor=bg_color,
        paper_bgcolor='white',
        font={'color': '#2d3748'},
        annotations=[
            dict(
                text="✨ Human-Centered Analysis",
                showarrow=False,
                xref="paper", yref="paper",
                x=1.0, y=1.02, xanchor='right', yanchor='bottom',
                font=dict(size=12, color='#718096')
            )
        ]
    )
    
    return fig

def code_interpret(e2b_code_interpreter: Sandbox, code: str) -> Optional[List[Any]]:
    with st.spinner('🤖 AI is crunching numbers...'):
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exec = e2b_code_interpreter.run_code(code)

        if stderr_capture.getvalue():
            print("[Code Interpreter Warnings/Errors]", file=sys.stderr)
            print(stderr_capture.getvalue(), file=sys.stderr)

        if stdout_capture.getvalue():
            print("[Code Interpreter Output]", file=sys.stdout)
            print(stdout_capture.getvalue(), file=sys.stdout)

        if exec.error:
            print(f"[Code Interpreter ERROR] {exec.error}", file=sys.stderr)
            return None
        return exec.results

def match_code_blocks(llm_response: str) -> str:
    match = pattern.search(llm_response)
    if match:
        code = match.group(1)
        return code
    return ""

def enhanced_chat_with_llm(e2b_code_interpreter: Sandbox, user_message: str, dataset_path: str, human_context: str = "") -> Tuple[Optional[List[Any]], str]:
    """Enhanced LLM chat that incorporates human context"""
    
    # Enhanced system prompt that considers human intelligence
    system_prompt = f"""You're a Python data scientist working alongside human intelligence. You have a dataset at '{dataset_path}' and the user's query.

IMPORTANT CONTEXT: The human has provided their intuition: "{human_context}"

Your job is to:
1. Analyze the dataset technically and accurately
2. Consider the human's intuitive insights
3. Look for patterns the human might have sensed but couldn't quantify
4. Generate Python code that explores both statistical patterns AND human behavioral factors
5. Always use the dataset path variable '{dataset_path}' when reading the CSV file
6. Create visualizations that tell a story, not just display data
7. Comment your code to explain what human insights you're investigating

Focus on finding the intersection between data science and human psychology."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    with st.spinner('🤖 AI is analyzing with human context...'):
        client = Together(api_key=st.session_state.together_api_key)
        response = client.chat.completions.create(
            model=st.session_state.model_name,
            messages=messages,
        )

        response_message = response.choices[0].message
        python_code = match_code_blocks(response_message.content)
        
        if python_code:
            code_interpreter_results = code_interpret(e2b_code_interpreter, python_code)
            return code_interpreter_results, response_message.content
        else:
            st.warning(f"Failed to match any Python code in model's response")
            return None, response_message.content

def upload_dataset(code_interpreter: Sandbox, uploaded_file) -> str:
    dataset_path = f"./{uploaded_file.name}"
    
    try:
        code_interpreter.files.write(dataset_path, uploaded_file)
        return dataset_path
    except Exception as error:
        st.error(f"Error during file upload: {error}")
        raise error

def load_sample_dataset(dataset_name: str) -> Tuple[pd.DataFrame, str]:
    """Load a sample dataset and return the dataframe and file path"""
    dataset_config = SAMPLE_DATASETS[dataset_name]
    file_path = dataset_config["path"]
    
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            # Handle Adidas dataset specifically - it has header issues
            if 'Adidas' in dataset_name:
                df = pd.read_excel(file_path, skiprows=3)
                # Clean up column names
                df.columns = ['Index', 'Retailer', 'Retailer ID', 'Invoice Date', 'Region', 
                             'State', 'City', 'Product', 'Price per Unit', 'Units Sold', 
                             'Total Sales', 'Operating Profit', 'Operating Margin', 'Sales Method']
                # Drop the index column and any empty rows
                df = df.dropna(subset=['Retailer']).reset_index(drop=True)
            else:
                df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        return df, file_path
    except Exception as error:
        st.error(f"Error loading sample dataset {dataset_name}: {error}")
        raise error

def upload_sample_dataset_to_sandbox(code_interpreter: Sandbox, dataset_name: str) -> str:
    """Upload a sample dataset to the sandbox and return the path"""
    dataset_config = SAMPLE_DATASETS[dataset_name]
    file_path = dataset_config["path"]
    
    try:
        # For Adidas dataset, we need to create a cleaned CSV version
        if 'Adidas' in dataset_name:
            # Load and clean the data
            df = pd.read_excel(file_path, skiprows=3)
            df.columns = ['Index', 'Retailer', 'Retailer ID', 'Invoice Date', 'Region', 
                         'State', 'City', 'Product', 'Price per Unit', 'Units Sold', 
                         'Total Sales', 'Operating Profit', 'Operating Margin', 'Sales Method']
            df = df.dropna(subset=['Retailer']).reset_index(drop=True)
            
            # Create a temporary CSV file
            temp_csv_path = "adidas_sales_cleaned.csv"
            df.to_csv(temp_csv_path, index=False)
            
            # Upload the cleaned CSV
            with open(temp_csv_path, 'rb') as file:
                file_content = file.read()
            
            sandbox_path = f"./adidas_sales_cleaned.csv"
            code_interpreter.files.write(sandbox_path, file_content)
            
            # Clean up temporary file
            os.remove(temp_csv_path)
            
            return sandbox_path
        else:
            # For other datasets, upload as is
            with open(file_path, 'rb') as file:
                file_content = file.read()
            
            # Upload to sandbox with the original filename
            filename = os.path.basename(file_path)
            sandbox_path = f"./{filename}"
            code_interpreter.files.write(sandbox_path, file_content)
            
            return sandbox_path
    except Exception as error:
        st.error(f"Error uploading sample dataset to sandbox: {error}")
        raise error

def main():
    """Main Streamlit application with Human Intelligence Layer"""
    
    # Custom CSS for better aesthetics
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .human-vs-ai {
        background: #f7fafc;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4299e1;
        margin: 1rem 0;
    }
    .insight-box {
        background: #fff5f5;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #fed7d7;
        margin: 0.5rem 0;
    }
    .human-input {
        background: #f0fff4;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #9ae6b4;
    }
    .sample-dataset-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .sample-dataset-card:hover {
        border-color: #4299e1;
        box-shadow: 0 4px 12px rgba(66, 153, 225, 0.15);
    }
    .dataset-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .key-insights {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .human-context-highlight {
        background: #dbeafe;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .technical-patterns {
        background: #f0f9ff;
        border: 2px solid #0ea5e9;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .human-insights {
        background: #fef3c7;
        border: 2px solid #f59e0b;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .pattern-insight-item {
        background: white;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        border-left: 3px solid #3b82f6;
    }
    .human-insight-item {
        background: white;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        border-left: 3px solid #f59e0b;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header"><h1> SoulAI: The Human Data Whisperer</h1><p>Where Human Intelligence Meets Artificial Intelligence</p></div>', unsafe_allow_html=True)
    
    st.markdown("""
    
    
    #### AI can crunch numbers perfectly, but can it understand the *human story* behind the data? 
    Let's find out where human intelligence still reigns supreme.
    
    ##### **🚀 How it works:**
    1. **You share your intuition** about the data (before AI sees it)
    2. **AI does its technical analysis** 
    3. **We compare and find** where human intelligence adds magic ✨
    """)

    # Initialize session state variables
    if 'together_api_key' not in st.session_state:
        st.session_state.together_api_key = ''
    if 'e2b_api_key' not in st.session_state:
        st.session_state.e2b_api_key = ''
    if 'model_name' not in st.session_state:
        st.session_state.model_name = ''
    if 'human_intuition' not in st.session_state:
        st.session_state.human_intuition = ''
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

    # Initialize Human Intelligence Layer
    hi_layer = HumanIntelligenceLayer()

    with st.sidebar:


                
        st.markdown("# CS Girlies Hackathon 2025")
        st.markdown("## 🎯 *The Challenge: AI vs H.I.*")

        st.markdown("---")
        st.header("🔧 Configuration")
        st.session_state.together_api_key = st.text_input("Together AI API Key", type="password")
        st.markdown("[Get Together AI API Key](https://api.together.ai/signin)")
        
        st.session_state.e2b_api_key = st.text_input("E2B API Key", type="password")
        st.markdown("[Get E2B API Key](https://e2b.dev/docs/legacy/getting-started/api-key)")
        
        # Model selection
        model_options = {
            "Meta-Llama 3.1 405B": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            "DeepSeek V3": "deepseek-ai/DeepSeek-V3",
            "Qwen 2.5 7B": "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "Meta-Llama 3.3 70B": "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        }
        selected_model = st.selectbox("Select AI Model", options=list(model_options.keys()), index=0)
        st.session_state.model_name = model_options[selected_model]
        


    # Dataset selection section
    st.markdown("## 📊 **Choose Your Dataset**")
    
    # Create tabs for different dataset options
    tab1, tab2 = st.tabs(["📁 Upload Your Own", "🎯 Try Sample Datasets"])
    
    df = None
    dataset_path = None
    selected_dataset_name = None
    
    with tab1:
        st.markdown("### Upload Your Own Dataset")
        uploaded_file = st.file_uploader("📁 Upload your dataset", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                else:
                    st.error("Unsupported file format. Please upload a CSV or Excel file.")
                    return
                
                dataset_path = uploaded_file.name
                st.success(f"✅ Successfully loaded: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error loading file: {e}")
                return
    
    with tab2:
        st.markdown("### 🎯 Try Our Sample Datasets")
        st.markdown("Don't have a dataset? No problem! Try one of our curated datasets:")
        
        # Use radio button for single selection
        dataset_options = list(SAMPLE_DATASETS.keys())
        selected_dataset_name = st.radio(
            "🎯 Choose a sample dataset:",
            options=dataset_options,
            format_func=lambda x: f"📊 {x}",
            key="sample_dataset_radio"
        )
        
        # Display detailed info for selected dataset
        if selected_dataset_name:
            config = SAMPLE_DATASETS[selected_dataset_name]
            st.markdown(f"""
            <div class="sample-dataset-card">
                <h4>📊 {selected_dataset_name}</h4>
                <p>{config['description']}</p>
                <p><strong>💡 Suggested Query:</strong> {config['preview_query']}</p>
                <p><strong>🧠 Human Context:</strong> {config['human_context_hint']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display key insights
            if 'key_insights' in config:
                insights_html = "<div class='key-insights'><h4>🔍 Key Human Insights to Look For:</h4><ul>"
                for insight in config['key_insights']:
                    insights_html += f"<li>{insight}</li>"
                insights_html += "</ul></div>"
                st.markdown(insights_html, unsafe_allow_html=True)
        
        # Load selected sample dataset
        if selected_dataset_name:
            try:
                df, dataset_path = load_sample_dataset(selected_dataset_name)
                st.markdown(f"""
                <div class="dataset-info">
                    <h3>🎉 Dataset Loaded Successfully!</h3>
                    <p><strong>Selected:</strong> {selected_dataset_name}</p>
                    <p><strong>File:</strong> {os.path.basename(dataset_path)}</p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error loading sample dataset: {e}")
                return
        
        if df is not None:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write("📊 **Your Dataset:**")
                show_full = st.checkbox("Show full dataset")
                if show_full:
                    st.dataframe(df)
                else:
                    st.write("Preview (first 5 rows):")
                    st.dataframe(df.head())
            
            with col2:
                st.write("📈 **Quick Stats:**")
                st.write(f"Rows: {len(df)}")
                st.write(f"Columns: {len(df.columns)}")
                st.write(f"Data types: {df.dtypes.nunique()}")
                
                # Show additional info for sample datasets
                if selected_dataset_name:
                    st.write("---")
                    st.write("🎯 **Sample Dataset Info:**")
                    st.write(f"Source: {selected_dataset_name}")
                    st.write(f"File: {os.path.basename(dataset_path)}")
        else:
            st.info("👆 Please select a dataset from the tabs above to get started!")
            return
        
        # Query input with auto-suggestion for sample datasets
        default_query = "What patterns or insights can we discover in this dataset?"
        if selected_dataset_name:
            default_query = SAMPLE_DATASETS[selected_dataset_name]["preview_query"]
        
        query = st.text_area(
            "🤔 What would you like to explore in your data?",
            default_query,
            height=100
        )
        
        # --- Human Intuition Capture Phase ---
        st.markdown("---")
        st.markdown("## 🧠 **Step 1: Your Human Intuition**")
        st.markdown("*Before we let AI analyze anything, share your human perspective:*")

        intuition_prompt = (
            "#### 🧠 HUMAN INTUITION CHECK\n"
            "Before we let AI analyze this data, what does YOUR human brain think?\n\n"
            "**Quick Data Preview:**"
        )
        st.markdown(intuition_prompt)
        st.code(df.head(3).to_string(index=False), language="text")

        st.markdown(f"**Your Question:** {query}")
        st.markdown("**What's your gut feeling?**")

        st.session_state.human_intuition = st.text_area(
            "Share your gut feeling, assumptions, or what you think the data might reveal:",
            placeholder="e.g., I think there might be seasonal patterns because this looks like sales data, and people probably buy more during holidays...",
            height=100,
            key="human_intuition_input"
        )

        # Human context suggestions
        human_suggestions = hi_layer.suggest_human_context(df, query)
        
        # Add sample dataset specific context
        if selected_dataset_name:
            config = SAMPLE_DATASETS[selected_dataset_name]
            sample_context = config["human_context_hint"]
            human_suggestions.insert(0, f"🎯 **Sample Dataset Context:** {sample_context}")
            
            # Add key insights from the sample dataset
            if 'key_insights' in config:
                human_suggestions.insert(1, "🔍 **Key Human Insights to Investigate:**")
                for insight in config['key_insights']:
                    human_suggestions.append(f"  • {insight}")
        
        if human_suggestions:
            st.markdown("**💡 Human Context Clues:**")
            for suggestion in human_suggestions:
                st.markdown(f"- {suggestion}")
        
        # Analysis button
        if st.button("🚀 Start AI vs Human Analysis", type="primary"):
            if not st.session_state.together_api_key or not st.session_state.e2b_api_key:
                st.error("Please enter both API keys in the sidebar.")
            elif not st.session_state.human_intuition:
                st.warning("Please share your human intuition first! That's the whole point of this experiment 😊")
            elif df is None:
                st.warning("Please select a dataset first! Choose from the sample datasets or upload your own.")
            else:
                st.markdown("---")
                st.markdown("## 🤖 **Step 2: AI Technical Analysis**")
                
                with Sandbox(api_key=st.session_state.e2b_api_key) as code_interpreter:
                    # Upload the dataset to sandbox
                    if selected_dataset_name:
                        # Use sample dataset
                        sandbox_dataset_path = upload_sample_dataset_to_sandbox(code_interpreter, selected_dataset_name)
                    else:
                        # Use uploaded file
                        sandbox_dataset_path = upload_dataset(code_interpreter, uploaded_file)
                    
                    # Enhanced AI analysis with human context
                    code_results, llm_response = enhanced_chat_with_llm(
                        code_interpreter, 
                        query, 
                        sandbox_dataset_path, 
                        st.session_state.human_intuition
                    )
                    
                    # AI Response
                    st.write("🤖 **AI's Technical Analysis:**")
                    st.write(llm_response)
                    
                    # Analyze emotional context
                    emotional_context = hi_layer.analyze_emotional_context(llm_response)
                    
                    # Display results/visualizations
                    if code_results:
                        st.markdown("### 📊 **Generated Visualizations:**")
                        for result in code_results:
                            if hasattr(result, 'png') and result.png:
                                png_data = base64.b64decode(result.png)
                                image = Image.open(BytesIO(png_data))
                                st.image(image, caption="AI-Generated Visualization", use_container_width=True)
                            elif hasattr(result, 'figure'):
                                fig = result.figure
                                st.pyplot(fig)
                            elif hasattr(result, 'show'):
                                st.plotly_chart(result)
                            elif isinstance(result, (pd.DataFrame, pd.Series)):
                                st.dataframe(result)
                            else:
                                st.write(result)
                    
                    # TECHNICAL PATTERNS vs HUMAN INSIGHTS COMPARISON
                    st.markdown("---")
                    st.markdown("## 🎯 **Step 3: Technical Patterns vs Human Insights**")
                    
                    # Extract technical patterns from AI response
                    technical_patterns = extract_technical_patterns(llm_response)
                    
                    # Extract human insights
                    human_insights = extract_human_insights(st.session_state.human_intuition, human_suggestions)
                    
                    # Display side-by-side comparison
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<div class="technical-patterns">', unsafe_allow_html=True)
                        st.markdown("### 🤖 **AI Technical Patterns**")
                        if technical_patterns:
                            for i, pattern in enumerate(technical_patterns, 1):
                                st.markdown(f"""
                                <div class="pattern-insight-item">
                                    <strong>{i}.</strong> {pattern}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No clear technical patterns identified by AI")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="human-insights">', unsafe_allow_html=True)
                        st.markdown("### 🧠 **Human Contextual Insights**")
                        if human_insights:
                            for i, insight in enumerate(human_insights, 1):
                                st.markdown(f"""
                                <div class="human-insight-item">
                                    <strong>{i}.</strong> {insight}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No unique human insights identified")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Create detailed comparison
                    comparison = create_human_vs_ai_comparison(
                        st.session_state.human_intuition,
                        llm_response,
                        emotional_context
                    )
                    
                    # Summary comparison
                    st.markdown("---")
                    st.markdown("## 📊 **Pattern vs Insight Summary**")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "🤖 Technical Patterns",
                            len(technical_patterns),
                            "AI Statistical Findings"
                        )
                    
                    with col2:
                        st.metric(
                            "🧠 Human Insights", 
                            len(human_insights),
                            "Contextual Understanding"
                        )
                    
                    with col3:
                        if len(technical_patterns) > 0 and len(human_insights) > 0:
                            ratio = len(human_insights) / len(technical_patterns)
                            if ratio > 1.5:
                                verdict = "Human Intelligence Dominates"
                                color = "success"
                            elif ratio > 0.7:
                                verdict = "Balanced Analysis"
                                color = "info"
                            else:
                                verdict = "AI Patterns Dominate"
                                color = "warning"
                        else:
                            verdict = "Insufficient Data"
                            color = "info"
                        
                        st.metric(
                            "🎯 Analysis Balance",
                            verdict,
                            f"{len(human_insights)}:{len(technical_patterns)} ratio"
                        )
                    
                    # Human Edge Analysis
                    st.markdown("---")
                    st.markdown("## 🎯 **The Human Edge: What AI Might Miss**")
                    
                    human_edge_insights = []
                    
                    # Analyze what AI might have missed
                    if len(human_insights) > len(technical_patterns):
                        human_edge_insights.append("🧠 **Contextual Understanding:** Humans excel at connecting data to real-world context")
                    
                    if any('emotion' in insight.lower() or 'feel' in insight.lower() for insight in human_insights):
                        human_edge_insights.append("💭 **Emotional Intelligence:** Humans understand emotional drivers behind data patterns")
                    
                    if any('culture' in insight.lower() or 'social' in insight.lower() for insight in human_insights):
                        human_edge_insights.append("🌍 **Cultural Awareness:** Humans recognize cultural and social factors affecting behavior")
                    
                    if any('behavior' in insight.lower() or 'psychology' in insight.lower() for insight in human_insights):
                        human_edge_insights.append("🧠 **Behavioral Psychology:** Humans understand motivation and decision-making processes")
                    
                    if any('trend' in insight.lower() or 'seasonal' in insight.lower() for insight in human_insights):
                        human_edge_insights.append("📅 **Temporal Context:** Humans recognize seasonal and temporal patterns in behavior")
                    
                    if not human_edge_insights:
                        human_edge_insights.append("🤖 **AI Dominance:** In this case, AI patterns may be more comprehensive than human insights")
                    
                    for insight in human_edge_insights:
                        st.markdown(f"""
                        <div class="human-context-highlight">
                            {insight}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('<div class="human-vs-ai">', unsafe_allow_html=True)
                    st.markdown(comparison)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Bias alerts
                    bias_alerts = hi_layer.generate_bias_alerts(llm_response)
                    if bias_alerts:
                        st.markdown("### ⚠️ **Human Intelligence Alerts:**")
                        for alert in bias_alerts:
                            st.markdown(f'<div class="insight-box">{alert}</div>', unsafe_allow_html=True)
                    
                    # --- Verdict Section ---
                    st.markdown("---")
                    st.markdown("## 🏆 **The Verdict: Where Humans Win**")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "🧠 Human Intuition Score",
                            "High" if len(st.session_state.human_intuition) > 50 else "Medium",
                            "Context & Emotion"
                        )
                        st.caption(
                            "Reflects the depth and nuance of your input. "
                            "A high score means you provided rich context, emotional cues, or cultural insights that go beyond the data."
                        )
                        # Show a snippet of the most insightful human input
                        if st.session_state.human_intuition:
                            st.markdown("**Example:**")
                            st.info(f"“{st.session_state.human_intuition[:120]}{'...' if len(st.session_state.human_intuition) > 120 else ''}”")

                    with col2:
                        st.metric(
                            "🤖 AI Technical Score", 
                            "High" if code_results else "Medium",
                            "Pattern Recognition"
                        )
                        st.caption(
                            "Shows how well the AI identified statistical patterns, trends, or correlations. "
                            "A high score means the AI found clear, data-driven insights."
                        )
                        # Extract and show main patterns/correlations from AI response
                        ai_patterns = re.findall(r"(correlation|trend|pattern|average|increase|decrease)", llm_response, re.IGNORECASE)
                        if ai_patterns:
                            st.markdown("**AI Found:**")
                            st.info(", ".join(set([p.capitalize() for p in ai_patterns])))
                        else:
                            st.markdown("**AI Found:**")
                            st.info("No major patterns or correlations detected.")

                    with col3:
                        st.metric(
                            "✨ Human Edge Factor",
                            f"{len(bias_alerts)} insights",
                            "What AI Missed"
                        )
                        st.caption(
                            "Counts the number of unique, human-contextual insights or bias alerts that the AI did not address. "
                            "A higher number means more areas where human intelligence added value."
                        )
                        # Show the actual bias alerts or human suggestions
                        if bias_alerts:
                            st.markdown("**Human Edge Details:**")
                            for alert in bias_alerts:
                                st.warning(alert)
                        else:
                            st.markdown("**Human Edge Details:**")
                            st.info("No unique human-contextual insights detected this time.")

                    # --- Comparison Summary ---
                    if 'ai_patterns' in locals() and len(bias_alerts) > len(set(ai_patterns)):
                        st.success("🎉 Human intelligence provided more unique insights than the AI's technical findings in this analysis!")
                    elif 'ai_patterns' in locals() and len(set(ai_patterns)) > len(bias_alerts):
                        st.info("🤖 The AI identified more technical patterns than unique human-contextual insights this time.")
                    else:
                        st.info("Both human and AI contributed equally valuable perspectives.")

                    # --- Hackathon Insight Section ---
                    st.markdown("### 🎯 **Hackathon Insight: AI vs H.I.**")
                    st.success(f"""
                    **The Human Intelligence Advantage:**
                    
                    While AI excelled at {llm_response.count('correlation') + llm_response.count('pattern')} technical patterns, 
                    human intelligence provided {len(bias_alerts)} contextual insights that AI completely missed.
                    
                    **Emotional Intelligence Score:** {emotional_context.get('emotional_tone', 'neutral').title()}  
                    *Indicates the overall emotional tone detected in the AI's analysis (positive, negative, or neutral).*
                    
                    **Human Context Factors:** {len(human_suggestions)} cultural/behavioral considerations  
                    *Represents the number of unique human context clues suggested for this dataset.*
                    
                    **The Bottom Line:** Humans don't just see data - we feel the story behind it. 🧠✨
                    """)
                    
                    st.session_state.analysis_complete = True

    # Demo section for hackathon judges
    if df is None:
        st.markdown("---")
        st.markdown("## 🎬 **Demo Preview : )**")
        
        demo_data = {
            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            'Sales': [1000, 1200, 800, 1500, 1800, 2200],
            'Customer_Satisfaction': [4.2, 4.1, 3.8, 4.5, 4.7, 4.9]
        }
        demo_df = pd.DataFrame(demo_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### **🤖 What AI Sees:**")
            st.markdown("""
            - Sales increased 120% from Jan to Jun
            - Strong positive correlation (r=0.89) between sales and satisfaction
            - Linear growth trend detected
            """)
        
        with col2:
            st.markdown("### **🧠 What Humans Feel:**")
            st.markdown("""
            - March dip = post-holiday blues? Budget constraints?
            - June spike = summer optimism? Bonus season?
            - Satisfaction follows sales = happy customers buy more (emotional connection)
            """)
        
        st.plotly_chart(
            px.line(demo_df, x='Month', y='Sales', title="📈 Demo: AI sees trends, Humans see stories"),
            use_container_width=True
        )

if __name__ == "__main__":
    main()