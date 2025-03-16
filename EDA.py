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
        analysis_insights = analyze_data_with_groq(client, df)
        prompt = f"""Generate business-focused Plotly visualizations for this dataset:
        
**Dataset Profile**
- Shape: {df_cleaned.shape}
- Numeric Columns: {list(df_cleaned.select_dtypes(include=np.number).columns)}
- Excluded Columns: {['Unnamed: 0', 'Order ID', 'Pizza ID']}

**Key Insights to Visualize**
{analysis_insights}

**Visualization Requirements**
1. Create 7-9 meaningful charts showing:
   - Time trends (line/area charts)
   - Price-quantity relationships (scatter/bubble)
   - Sales distributions (histogram/density)
   - Product/category breakdowns (bar/pie)
   - Hourly patterns (heatmap/facetted line)

2. Forbidden:
   - Box plots
   - ID columns
   - Unaggregated high-cardinality data

3. Each chart MUST:
   - Use Plotly Express
   - Start with "# Fig [N]: [Business Insight]" comment
   - Include data aggregations where needed
   - Have proper titles/axis labels with units
   - Use color strategically
   - Include st.plotly_chart(fig, use_container_width=True)

**Example Good Visualization**
```python
# Fig 1: Monthly Revenue Trend with 3-Month Moving Average
monthly_data = df_cleaned.groupby('Month', as_index=False)['Sales'].sum()
monthly_data['3MMA'] = monthly_data['Sales'].rolling(3).mean()
fig = px.line(monthly_data, x='Month', y=['Sales', '3MMA'], 
             title='Monthly Sales Trend with Moving Average',
             labels={{'value': 'Revenue (USD)'}})
fig.update_layout(legend_title_text='Metric')
st.plotly_chart(fig, use_container_width=True)
Dataset Sample
{df_cleaned.head(3).to_string()}
"""

        completion = client.chat.completions.create(
            model="qwen-2.5-coder-32b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=131072
        )

        response = completion.choices[0].message.content
        code_blocks = re.findall(r'```python(.*?)```', response, re.DOTALL)

        if not code_blocks:
            st.error("No valid code found in response")
            return

        exec_globals = {
            'pd': pd, 'np': np, 'st': st,
            'px': px, 'go': go, 'ff': ff,
            'df_cleaned': df_cleaned  # Pass the actual dataframe
        }

        st.subheader("Full Data Visualizations")
        for code in code_blocks:
            try:
                st.code(code.strip(), language='python')
                # Execute the code which should contain st.plotly_chart() calls
                exec(code.strip(), exec_globals)
            except Exception as e:
                st.error(f"Error executing code block: {str(e)}")

    except Exception as e:
        st.error(f"Visualization error: {str(e)}")


data = st.file_uploader("Upload a CSV file", type=["csv"])

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
