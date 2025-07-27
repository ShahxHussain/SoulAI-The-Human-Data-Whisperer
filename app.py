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
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header"><h1>🧠 SoulAI: The Human Data Whisperer</h1><p>Where Human Intelligence Meets Artificial Intelligence</p></div>', unsafe_allow_html=True)
    
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
        


    # File upload
    uploaded_file = st.file_uploader("📁 Upload your dataset", type="csv")
    
    if uploaded_file is not None:
        # Load and display dataset
        df = pd.read_csv(uploaded_file)
        
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
        
        # Query input
        query = st.text_area(
            "🤔 What would you like to explore in your data?",
            "What patterns or insights can we discover in this dataset?",
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
            else:
                st.markdown("---")
                st.markdown("## 🤖 **Step 2: AI Technical Analysis**")
                
                with Sandbox(api_key=st.session_state.e2b_api_key) as code_interpreter:
                    # Upload the dataset
                    dataset_path = upload_dataset(code_interpreter, uploaded_file)
                    
                    # Enhanced AI analysis with human context
                    code_results, llm_response = enhanced_chat_with_llm(
                        code_interpreter, 
                        query, 
                        dataset_path, 
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
                    
                    # HUMAN VS AI COMPARISON
                    st.markdown("---")
                    st.markdown("## 🎯 **Step 3: The Human Edge Analysis**")
                    
                    # Create comparison
                    comparison = create_human_vs_ai_comparison(
                        st.session_state.human_intuition,
                        llm_response,
                        emotional_context
                    )
                    
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
    if not uploaded_file:
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