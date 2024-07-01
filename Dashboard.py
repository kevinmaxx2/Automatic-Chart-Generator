import streamlit as st
import pandas as pd
import plotly.express as px
import logging
import base64
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler

# Initialize session state
def init_session_state():
    if "theme" not in st.session_state:
        st.session_state.theme = "light"

    if "selected_columns" not in st.session_state:
        st.session_state.selected_columns = []

    if "filter_conditions" not in st.session_state:
        st.session_state.filter_conditions = ""

# Toggle theme function
def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

# Apply theme function
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

# Function to load file and return DataFrame
@st.cache_data
def load_file(uploaded_file):
    try:
        if uploaded_file.type == 'text/csv':
            with st.spinner('Processing CSV file...'):
                return pd.read_csv(uploaded_file)
        elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            with st.spinner('Processing Excel file...'):
                return pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty.")
    except pd.errors.ParserError:
        st.error("The uploaded file contains invalid data.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Function to display file preview
def show_file_preview(df):
    st.write("File Preview:")
    st.write(df.head())
    st.write("Data Summary:")
    st.write(df.describe())

# Function to process data (remove missing values and normalize if selected)
def process_data(df, numeric_columns):
    if st.checkbox('Remove missing values'):
        df = df.dropna()
    if st.checkbox('Normalize data'):
        scaler = MinMaxScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df

# Function to filter data based on selected columns and filter conditions
def filter_data(df):
    st.sidebar.header("Filter Data")
    columns = df.columns.tolist()
    columns_to_display = [col for col in columns if col not in st.session_state.selected_columns]
    
    selected_columns = st.sidebar.multiselect("Select columns to include:", columns_to_display, default=st.session_state.selected_columns)
    filter_conditions = st.sidebar.text_area("Enter filter conditions (e.g., column_name > 50):", value=st.session_state.filter_conditions)
    
    if filter_conditions:
        st.session_state.filter_conditions = filter_conditions
    
    st.session_state.selected_columns = selected_columns
    
    if st.session_state.selected_columns:
        df = df[selected_columns]
    
    if st.session_state.filter_conditions:
        try:
            df = df.query(st.session_state.filter_conditions)
        except pd.errors.ParserError:
            st.error("Invalid filter conditions. Please check your input.")
    
    return df

# Function to plot data based on user selections
def plot_data(df, index_column, numeric_columns):
    column_choice = st.selectbox("Select the column for the chart:", numeric_columns)
    chart_choice = st.radio("Choose your graph type:", ('Line Chart', 'Bar Chart', 'Scatter Plot'))
    agg_choice = st.radio("Choose the aggregation type:", ('Average', 'Count'))

    try:
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

        # Descriptive Statistics
        st.subheader("Descriptive Statistics")

        # Display Skewness and Kurtosis side by side
        stats_columns = st.columns(2)
        stats_columns[0].write("**Skewness:**")
        stats_columns[0].write(df[numeric_columns].apply(lambda x: skew(x)))
        stats_columns[1].write("**Kurtosis:**")
        stats_columns[1].write(df[numeric_columns].apply(lambda x: kurtosis(x)))

        # Correlation Matrix    
        st.subheader("Correlation Matrix")
        
        # Filter numeric columns for correlation calculation
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        
        if not numeric_df.empty:
            corr_matrix = numeric_df.corr()
            st.write(corr_matrix)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5, ax=ax)
            st.pyplot(fig)
        else:
            st.write("No numeric columns available to calculate correlation.")

    except Exception as e:
        st.error(f"An error occurred while generating the chart: {e}")

# Function to generate a download link for the processed data
def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  
    href = f'<a href="data:file/csv;base64,{b64}" download="processed_data.csv">Download CSV file</a>'
    return href

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Main function to run the app
def main():
    init_session_state()

    # Apply theme
    apply_theme()

    # Add the theme toggle button in the sidebar
    button_label = "Switch to Dark Mode" if st.session_state.theme == "light" else "Switch to Light Mode"
    st.sidebar.button(button_label, on_click=toggle_theme)

    st.title('Automatic Dashboard')

    file = st.file_uploader('Insert your file here!', type=['csv', 'xlsx'])

    if file is not None:
        with st.spinner('Uploading file...'):
            df = load_file(file)

        if df is not None:
            st.success("File uploaded successfully!")
            if len(df) > 10000:
                st.warning("The uploaded file has more than 10,000 rows. Displaying only the first 10,000 rows.")
                df = df.head(10000)
            all_columns = df.columns.tolist()
            index_column = st.selectbox("Select the index column:", all_columns)
            numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != index_column]

            if numeric_columns:
                df = filter_data(df)
                filtered_numeric_columns = [col for col in numeric_columns if col in df.columns]
                df = process_data(df, filtered_numeric_columns)
                st.markdown(get_table_download_link(df), unsafe_allow_html=True)
                plot_data(df, index_column, filtered_numeric_columns)
            else:
                st.error("No numeric columns in your data")
    else:
        st.warning('No file uploaded')

    if file:
        logger.info("File uploaded: %s", file.name)

if __name__ == "__main__":
    main()