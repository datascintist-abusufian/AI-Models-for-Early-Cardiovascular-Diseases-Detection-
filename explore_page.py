import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

@st.cache_data
def load_data():
    df = pd.read_csv('heart.csv')
    return df

# Load the data
df = load_data()
corr = df.corr() 

def show_explore_page():
    st.title("Explore Heart Disease Data")

    st.write(
        """
        ### Heart Disease Data Plots.
        """
    )

    st.title("Target Variable Visualization")

    # Create a Figure and Axes using Matplotlib
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.set(style="whitegrid")
    sns.countplot(x="target", data=df, ax=ax2)
    ax2.set_xlabel("Target Variable")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of Target Variable")

    # Display the Figure and Axes in the Streamlit app
    st.pyplot(fig2)


    
    st.subheader('Heart Disease Frequency for Slope compared to each gender')
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.barplot(x='slope', y='target', hue='sex', data=df, palette=['#DAF7A6', '#FF5733'], ax=ax)
    ax.set_title('Heart Disease Frequency for Slope')
    ax.set_xlabel('The Slope of The Peak Exercise ST Segment')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_ylabel('Frequency')

    # Display the plot on Streamlit
    st.pyplot(fig)

    fig1 = go.Figure(data=go.Heatmap(
        z=corr.values,  # Extract the values from the correlation DataFrame
        x=corr.index.values,
        y=corr.columns.values,
        colorscale='earth',
    ))

    fig1.update_layout(
        title_text='<b>Correlation Matrix (cont. features)<b>',
        title_x=0.5,
        titlefont={'size': 24},
        width=550,
        height=550,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange='reversed',
        paper_bgcolor=None,
    )

    st.title("Correlation Heatmap")
    # Display the heatmap in the Streamlit app
    st.plotly_chart(fig1)


    

if __name__ == '__main__':
    show_explore_page()

 