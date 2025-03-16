import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
from groq import Groq
import re
# from pandasai import SmartDataframe
# from pandasai.connectors import PandasConnector
import plotly.graph_objects as go
import dash

def initialize_groq_client():
    try:
        client = Groq(api_key="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")
        return client
    except Exception as e:
        st.error(f"An error occurred while initializing Groq client: {str(e)}")
        return None
    
def load_data(data):
    if data.name.endswith('.csv'):
        df = pd.read_csv(data)
    return df

def sanitize_dataframe(df):
    if df is not None:
        numeric_df = df.select_dtypes(include=[np.number])
        return numeric_df

def summarize_dataframe(df):
    if df is not None:
        summary = df.describe(include='all').to_json(orient='records')
        return summary

def analyze_data_with_groq(client, df):
    try:
        df_cleaned = sanitize_dataframe(df)
        if df_cleaned.empty:
            return ""

        summary = summarize_dataframe(df_cleaned)
        
        completion = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{
                "role": "user",
                "content": f"""Analyze this data summary and provide:
                1. Key Facts (bullet points of essential metrics)
                2. Recommendations (actionable insights)
                3. Conclusion (final observation)
            
                Summary: {summary}"""
            }],
            temperature=0.6,
            max_tokens=131072,
            top_p=0.9
        )

        raw_output = completion.choices[0].message.content
        cleaned_output = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL)
        
        return cleaned_output.strip()

    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return ""

def visualize_data_with_groq(client, df):
    try:
        df_cleaned = sanitize_dataframe(df)
        if df_cleaned.empty:
            return

        # First get analysis insights
        analysis_insights = analyze_data_with_groq(client, df)
        
        # Enhanced prompt with explicit variable names
        prompt = f"""Generate Plotly visualizations using THIS EXACT DATAFRAME NAME: df_cleaned
Dataset shape: {df_cleaned.shape}
Columns: {', '.join(df_cleaned.columns)}
First 3 rows:
{df_cleaned.head(3).to_string()}

Requirements:
1. Use ONLY these columns: {[c for c in df_cleaned.columns if 'id' not in c.lower()]}
2. NEVER use these columns: {[c for c in df_cleaned.columns if 'id' in c.lower()]}
3. Each visualization MUST:
   - Start with '# Fig [N]: [Meaningful Title]' comment
   - Use df_cleaned (pre-filtered numeric data)
   - Include st.plotly_chart(fig, use_container_width=True)
   - Have proper axis labels with units
4. Generate 5-7 visualizations of different types
5. Focus on meaningful business insights from: {analysis_insights}

Example Code:
```python
# Fig 1: Monthly Sales Trend
monthly_sales = df_cleaned.groupby('Month', as_index=False)['Sales'].sum()
fig = px.line(monthly_sales, x='Month', y='Sales', 
             title='Monthly Sales Trend')
fig.update_layout(xaxis_title='Month', yaxis_title='Total Sales (USD)')
st.plotly_chart(fig, use_container_width=True)
```"""

        completion = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4096
        )

        response = completion.choices[0].message.content
        code_blocks = re.findall(r'```python(.*?)```', response, re.DOTALL)

        if not code_blocks:
            st.error("No valid code found in response")
            return

        # Create execution context with ACTUAL dataframe
        exec_globals = {
            'pd': pd, 'np': np, 'st': st,
            'px': px, 'go': go, 'ff': ff,
            'df_cleaned': df_cleaned  # Pass cleaned dataframe with this name
        }

        st.subheader("Insightful Visualizations")
        for idx, code in enumerate(code_blocks, 1):
            try:
                with st.expander(f"Visualization {idx}", expanded=True):
                    st.code(code.strip(), language='python')
                    # Execute in isolated context
                    exec(code.strip(), exec_globals)
            except Exception as e:
                st.error(f"""Error in Visualization {idx}:
                {str(e)}
                Code snippet: {code[:200]}...""")

    except Exception as e:
        st.error(f"Visualization error: {str(e)}")

if data is not None:
    df = load_data(data)

    if df is not None:
        st.write("Data Preview:")
        st.write(df.head())

        client = initialize_groq_client()
        if client is not None:
            st.write("Analyzing data with Groq...")
            analysis = analyze_data_with_groq(client, df)
            st.write("Analysis Report:")
            st.write(analysis)

            st.write("Generating Visualization Report...")
            visualization = visualize_data_with_groq(client, df)
            st.write("Visualization Report:")
            st.write(visualization)
