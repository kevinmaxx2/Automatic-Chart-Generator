# Automatic Dashboard

Automatic Dashboard is a Streamlit web application designed to facilitate data analysis and visualization from uploaded CSV or Excel files. It provides functionalities for data preview, filtering, normalization, and dynamic chart generation based on user selections.

## Features

- **File Upload**: Upload CSV or Excel files for analysis.
- **Data Preview**: View the first few rows and summary statistics of the uploaded dataset.
- **Filtering**: Select specific columns and apply custom filter conditions to refine data.
- **Normalization**: Normalize numeric data columns using Min-Max scaling.
- **Chart Generation**: Generate line charts, bar charts, or scatter plots based on selected columns and aggregation types (average or count).
- **Descriptive Statistics**: Display skewness, kurtosis, and correlation matrix for numeric columns.

## Getting Started

### Prerequisites

Make sure you have Python and pip installed. Install necessary packages using:

    pip install -r requirements.txt
### Running the App

Run the Streamlit app with:

    streamlit run Dashboard.py

## Usage

1. **Upload File**: Click on the file uploader and select a CSV or Excel file.
2. **Data Analysis**:
   - Select the index column for data indexing.
   - Choose columns for filtering and specify filter conditions (optional).
   - Normalize data if needed.
   - Select the type of chart and aggregation method.
3. **Visualization**:
   - View generated charts based on selected criteria.
   - Explore descriptive statistics and correlation matrix.
4. **Download Data**: Download processed data as a CSV file.

### Theme Customization

- The app supports light and dark themes. Use the theme toggle button in the sidebar to switch between themes.

## Contributing

Contributions are welcome! If you have suggestions, improvements, or bug fixes, please submit an issue or pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built using Streamlit, Pandas, Plotly, Seaborn, and other open-source libraries.
- Inspired by data analysis and visualization needs in various domains.
