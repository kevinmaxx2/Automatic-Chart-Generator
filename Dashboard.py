import streamlit as st
import pandas as pd
import plotly.express as px
import time
import logging
from sklearn.preprocessing import MinMaxScaler


# Initialize session state for theme if not already set
if "theme" not in st.session_state:
    st.session_state.theme = "light"

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

def apply_theme():
    common_styles = """
    <style>
    .main { padding: 10px; }
    .stButton>button { border-radius: 5px; }
    </style>
    """
    light_theme = """
    <style>
    .main { background-color: #FFFFFF; color: #000000; font-size: 16px; }
    .stTitle h1 { color: #000000; font-size: 32px; }
    .stButton>button { background-color: #FFFFFF; color: #000000; }
    </style>
    """
    dark_theme = """
    <style>
    .main { background-color: #121212; color: #FFFFFF; font-size: 16px; }
    .stTitle h1 { color: #D3D3D3; font-size: 32px; }
    .stButton>button { background-color: #4CAF50; color: #FFFFFF; }
    .stFileUploader label, .stRadio div, .stSelectbox div { color: #FFFFFF; }
    .stSelectbox select, .stSelectbox div { color: #FFFFFF; background-color: #2E2E2E; }
    </style>
    """
    st.markdown(common_styles, unsafe_allow_html=True)
    if st.session_state.theme == "dark":
        st.markdown(dark_theme, unsafe_allow_html=True)
    else:
        st.markdown(light_theme, unsafe_allow_html=True)

def load_file(uploaded_file):
    try:
        if uploaded_file.type == 'text/csv':
            with st.spinner('Processing file...'):
                return pd.read_csv(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a CSV file.")
    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty.")
    except pd.errors.ParserError:
        st.error("The uploaded file contains invalid data.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        logging.exception("Exception occurred while loading file")
    return None

def show_file_preview(df):
    st.write("File Preview:")
    st.write(df.head())
    st.write("Data Summary:")
    st.write(df.describe())

def process_data(df, numeric_columns):
    if st.checkbox('Remove missing values'):
        df = df.dropna()
    if st.checkbox('Normalize data'):
        scaler = MinMaxScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df

def plot_data(df, index_column, numeric_columns):
    column_choice = st.selectbox("Select the column for the chart:", numeric_columns)
    chart_choice = st.radio("Choose your graph type:", ('Line Chart', 'Bar Chart', 'Scatter Plot'))
    agg_choice = st.radio("Choose the aggregation type:", ('Average', 'Count'))

    # Aggregate data based on user selection
    if agg_choice == 'Average':
        data = df.set_index(index_column)[column_choice].groupby(index_column).mean()
    elif agg_choice == 'Count':
        data = df.set_index(index_column)[column_choice].groupby(index_column).count()

    # Plot the chart based on user selection
    if chart_choice == 'Line Chart':
        st.line_chart(data)
    elif chart_choice == 'Bar Chart':
        st.bar_chart(data)
    elif chart_choice == 'Scatter Plot':
        st.write(px.scatter(df, x=index_column, y=column_choice))
    
    # Display chart information
    st.subheader("Chart Information")
    st.write(f"**Chart Type:** {chart_choice}")
    st.write(f"**Column Selected:** {column_choice}")
    st.write(f"**Aggregation Type:** {agg_choice}")
    st.write(f"**Data Points:** {len(data)}")

    if chart_choice == 'Scatter Plot':
        st.write(f"**X-axis:** {index_column}")
        st.write(f"**Y-axis:** {column_choice}")
    
    st.write("**Data Types:**")
    st.write(df.dtypes)
    
    st.write("**Data Summary:**")
    st.write(df.describe())

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Apply theme
apply_theme()

# Add the theme toggle button in the sidebar
button_label = "Switch to Dark Mode" if st.session_state.theme == "light" else "Switch to Light Mode"
st.sidebar.button(button_label, on_click=toggle_theme)

st.title('Automatic Dashboard')

file = st.file_uploader('Insert your file here!', type=['csv'])

if file:
    st.write("Uploading file...")
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)
        progress_bar.progress(percent_complete + 1)

    df = load_file(file)
    if df is not None:
        if len(df) > 10000:
            st.warning("The uploaded file has more than 10,000 rows. Displaying only the first 10,000 rows.")
            df = df.head(10000)
        all_columns = df.columns.tolist()
        index_column = st.selectbox("Select the index column:", all_columns)
        numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != index_column]

        if numeric_columns:
            df = process_data(df, numeric_columns)
            plot_data(df, index_column, numeric_columns)
        else:
            st.error("No numeric columns in your data")
else:
    st.warning('No file uploaded')

if file:
    logger.info("File uploaded: %s", file.name)
