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
                2. Recommendations (actionable insights) (example : Promote Bulk Purchases, Increase Marketing Spend,Tax related facts and many other things)
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

        # Enhanced prompt with strict dataset usage instructions
        prompt = f"""Generate visualizations using THE ACTUAL DATASET PROVIDED (df_cleaned).
        DO NOT CREATE OR MODIFY THE DATASET. Use these columns: {list(df_cleaned.columns)}
        
        Requirements:
        1. Use ONLY this data: df_cleaned (shape: {df_cleaned.shape})
        2. Forbidden:
           - Any pd.DataFrame() creations
           - Hardcoded data
           - Example/test data
           - Boxplots
        3. Required visualizations:
           - Temporal trends (line/area charts)
           - Correlation analysis (scatter plots)
           - Distribution analysis (histograms)
           - Categorical breakdowns (bar/pie)
           - Hourly patterns (heatmaps)
           - Include many other visualization like geographic when longitude and latitude is present (use st.map(map_data))
           - Use Most of the columns
        4. Create 8 different chart/plots types focusing on these relationships
        5. OUTPUT FORMAT:
           - Only Python code within ```python blocks
           - One visualization per code block
           - Include necessary aggregations
        6. Each visualization MUST:
            * Use Plotly Express
            * Have meaningful title starting with "Fig [N]: "
            * Include axis labels with units
            * Contain <50 words caption in # comments explaining insight
            * Use st.plotly_chart() with full width
        
        Example VALID code:
        ```python
        # Fig 1: Sales distribution by month
        monthly_sales = df_cleaned.groupby('Month', as_index=False)['Sales'].sum()
        fig = px.bar(monthly_sales, x='Month', y='Sales', 
                    title='Actual Monthly Sales from Dataset')
        st.plotly_chart(fig, use_container_width=True)
        ```
        
        First 3 rows of ACTUAL DATA:
        {df_cleaned.head(3).to_string()}
        """

        completion = client.chat.completions.create(
            model="qwen-2.5-coder-32b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Lower temperature for less creativity
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
            'df_cleaned': df_cleaned  # Pass actual dataframe
        }

        st.subheader("Full Dataset Visualizations")
        for idx, code in enumerate(code_blocks, 1):
            try:
                # Validate code contains actual dataset reference
                if 'df_cleaned' not in code:
                    st.error(f"Visualization {idx} rejected: No dataset reference")
                    continue
                    
                if 'pd.DataFrame(' in code:
                    st.error(f"Visualization {idx} rejected: Creates new dataframe")
                    continue

                with st.expander(f"Visualization {idx}: Code", expanded=False):
                    st.code(code.strip(), language='python')
                
                # Execute in controlled environment
                exec(code.strip(), exec_globals)
                
                # # Force display if figure wasn't shown
                # if 'fig' in exec_globals:
                #     st.plotly_chart(exec_globals['fig'], use_container_width=True)

            except Exception as e:
                st.error(f"Error in visualization {idx}: {str(e)}")

    except Exception as e:
        st.error(f"Visualization error: {str(e)}")


def handle_custom_query(client, df_cleaned, query):
    """Handle custom user queries and generate visualizations"""
    try:
        prompt = f"""Generate a visualization and explanation based on this query:
        Query: {query}
        
        Requirements:
        1. Dataset Columns: {list(df_cleaned.columns)}
        2. Use ONLY df_cleaned (shape: {df_cleaned.shape})
        3. Output format:
           ## Explanation:
           [Textual explanation of the insight and recommendations]
           
           ## Code:
           ```python
           [Plotly visualization code using df_cleaned]
           ```
        4. Code must:
           - Use Plotly Express
           - Include proper labels/titles
           - Use st.plotly_chart()
        """

        completion = client.chat.completions.create(
            model="qwen-2.5-coder-32b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=131072
        )

        response = completion.choices[0].message.content
        
        # Extract explanation and code
        explanation = re.search(r'## Explanation:(.*?)(## Code:|\Z)', response, re.DOTALL)
        code = re.search(r'```python(.*?)```', response, re.DOTALL)
        
        return (
            explanation.group(1).strip() if explanation else "No explanation generated",
            code.group(1).strip() if code else None
        )

    except Exception as e:
        st.error(f"Query processing error: {str(e)}")
        return None, None

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
            st.divider()
            st.subheader("Custom Data Query")
            
            query = st.text_area("Ask a question about your data (e.g., 'Show sales trends by month'):")
            
            if st.button("Generate Custom Visualization"):
                with st.spinner("Processing your query..."):
                    df_cleaned = sanitize_dataframe(df)
                    explanation, code = handle_custom_query(client, df_cleaned, query)
                    
                    if explanation:
                        st.subheader("Insight Explanation")
                        st.write(explanation)
                    
                    if code:
                        st.subheader("Generated Visualization")
                        try:
                            exec_globals = {
                                'pd': pd, 'np': np, 'st': st,
                                'px': px, 'go': go, 'ff': ff,
                                'df_cleaned': df_cleaned
                            }
                            exec(code, exec_globals)
                        except Exception as e:
                            st.error(f"Error executing custom visualization: {str(e)}")
                            st.code(code, language='python')
                    else:
                        st.error("Could not generate visualization for this query")
