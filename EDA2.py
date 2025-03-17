import base64
import io
from PIL import Image
from fpdf import FPDF
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

# Function to convert a figure to an image
def fig_to_image(fig):
    img_bytes = fig.to_image(format="png")
    return Image.open(io.BytesIO(img_bytes))

# Function to create a download link for a file
def get_download_link(file, filename, text):
    b64 = base64.b64encode(file).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Function to generate an HTML file with all content
def generate_html_file(content):
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Streamlit Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2e86c1; }}
            .visualization {{ margin-bottom: 30px; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>Streamlit Report</h1>
        {content}
    </body>
    </html>
    """
    return html_content.encode()

# Modify the analyze_data_with_groq function to include HTML download
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
        
        # Display the analysis report
        st.write("Analysis Report:")
        st.write(cleaned_output)
        
        # Generate HTML content for the report
        html_content = f"""
        <h2>Analysis Report</h2>
        <pre>{cleaned_output}</pre>
        """
        
        # Add a download button for the HTML file
        html_file = generate_html_file(html_content)
        st.markdown(get_download_link(html_file, "analysis_report.html", "Download Full Report (HTML)"), unsafe_allow_html=True)
        
        return cleaned_output.strip()

    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return ""

# Modify the visualize_data_with_groq function to include HTML download
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
           - Box plots
        3. Required visualizations:
           - Temporal trends (line/area charts)
           - Correlation analysis of Columns (using One scatter plots or heatmap is must)
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
            * Represents columns using colors in ONLY scatter plots
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
        visualization_images = []  # Store images for HTML content
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
                
                # Add a download button for the visualization
                if 'fig' in exec_globals:
                    fig = exec_globals['fig']
                    img = fig_to_image(fig)
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    st.markdown(get_download_link(byte_im, f"visualization_{idx}.png", f"Download Visualization {idx} (PNG)"), unsafe_allow_html=True)
                    
                    # Save the image for HTML content
                    img_base64 = base64.b64encode(byte_im).decode()
                    visualization_images.append(f'<div class="visualization"><h3>Visualization {idx}</h3><img src="data:image/png;base64,{img_base64}" alt="Visualization {idx}"></div>')

            except Exception as e:
                st.error(f"Error in visualization {idx}: {str(e)}")

        # Generate HTML content for all visualizations
        if visualization_images:
            html_content = "<h2>Visualizations</h2>" + "".join(visualization_images)
            html_file = generate_html_file(html_content)
            st.markdown(get_download_link(html_file, "visualizations.html", "Download All Visualizations (HTML)"), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Visualization error: {str(e)}")
