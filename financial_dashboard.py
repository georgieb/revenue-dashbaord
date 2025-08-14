import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
import os

# Page configuration
st.set_page_config(
    page_title="Financial Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_data_from_file(file_path=None, uploaded_file=None):
    """Load data from CSV or Excel file"""
    try:
        if uploaded_file is not None:
            # Handle uploaded file
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding='latin-1')
                    except:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding='cp1252')
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            else:
                st.error("❌ Unsupported file format. Please use CSV or Excel files.")
                return None
            
            st.success(f"✅ Successfully loaded {len(df)} records from uploaded file")
            
        elif file_path and os.path.exists(file_path):
            # Handle local file path
            file_extension = file_path.split('.')[-1].lower()
            
            if file_extension == 'csv':
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(file_path, encoding='latin-1')
                    except:
                        df = pd.read_csv(file_path, encoding='cp1252')
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(file_path)
            else:
                st.error("❌ Unsupported file format. Please use CSV or Excel files.")
                return None
            
            st.success(f"✅ Successfully loaded {len(df)} records from {file_path}")
        else:
            st.error("❌ No valid data source provided")
            return None
        
        # Clean and process the data
        df_cleaned = clean_data(df)
        display_data_info(df_cleaned)
        return df_cleaned
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

def clean_data(df):
    """Clean and standardize the data"""
    df_cleaned = df.copy()
    
    # Column mapping for common variations (case-insensitive)
    column_mapping = {
        'date': 'Date', 'transaction_date': 'Date', 'order_date': 'Date', 'invoice_date': 'Date',
        'amount': 'Amount', 'revenue': 'Amount', 'total': 'Amount', 'value': 'Amount', 'sales': 'Amount',
        'client': 'Client name', 'customer': 'Client name', 'customer_name': 'Client name', 'client_name': 'Client name',
        'company': 'Company', 'entity': 'Company', 'business_unit': 'Company', 'division': 'Company',
        'company_type': 'Company Type', 'business_type': 'Company Type', 'segment': 'Company Type',
        'class': 'Account Class', 'account_class': 'Account Class', 'tier': 'Account Class', 'category': 'Account Class',
        'account_mapping': 'Account mapping', 'account_type': 'Account mapping', 'mapping': 'Account mapping',
        'city': 'Ship To City', 'state': 'Ship To State', 'zip': 'Ship Zip', 'zip_code': 'Ship Zip',
    }
    
    # Apply column mapping (case-insensitive)
    for old_col in df_cleaned.columns:
        old_col_lower = old_col.lower().strip()
        if old_col_lower in column_mapping:
            new_col = column_mapping[old_col_lower]
            df_cleaned = df_cleaned.rename(columns={old_col: new_col})
            st.info(f"ℹ️ Mapped column '{old_col}' → '{new_col}'")
    
    # Fix data types
    df_cleaned = fix_data_types(df_cleaned)
    
    # Add missing columns with defaults
    df_cleaned = add_missing_columns(df_cleaned)
    
    return df_cleaned

def fix_data_types(df):
    """Fix data types for key columns"""
    df_fixed = df.copy()
    
    # Fix Date column
    if 'Date' in df_fixed.columns:
        try:
            df_fixed['Date'] = pd.to_datetime(df_fixed['Date'], errors='coerce')
            # Remove rows with invalid dates
            initial_count = len(df_fixed)
            df_fixed = df_fixed.dropna(subset=['Date'])
            if len(df_fixed) < initial_count:
                st.warning(f"⚠️ Removed {initial_count - len(df_fixed)} rows with invalid dates")
        except:
            st.warning("⚠️ Could not parse Date column automatically")
    
    # Fix Amount column
    if 'Amount' in df_fixed.columns:
        try:
            # Remove currency symbols and commas if it's text
            if df_fixed['Amount'].dtype == 'object':
                df_fixed['Amount'] = df_fixed['Amount'].astype(str)
                df_fixed['Amount'] = df_fixed['Amount'].str.replace('[$,€£¥]', '', regex=True)
                df_fixed['Amount'] = df_fixed['Amount'].str.replace('[(),]', '', regex=True)
            
            df_fixed['Amount'] = pd.to_numeric(df_fixed['Amount'], errors='coerce')
            # Remove rows with invalid amounts
            initial_count = len(df_fixed)
            df_fixed = df_fixed.dropna(subset=['Amount'])
            if len(df_fixed) < initial_count:
                st.warning(f"⚠️ Removed {initial_count - len(df_fixed)} rows with invalid amounts")
        except:
            st.warning("⚠️ Could not parse Amount column automatically")
    
    # Create Month column if Date exists
    if 'Date' in df_fixed.columns:
        df_fixed['Month'] = df_fixed['Date'].dt.strftime('%Y-%m')
    
    return df_fixed

def add_missing_columns(df):
    """Add missing columns with default values"""
    df_with_defaults = df.copy()
    
    # Required columns with defaults (only add if completely missing)
    required_columns = {
        'Company Type': 'Standard',
        'Client name': 'Unknown Client',
        'WTB Account': 'Sales',
        'Ship To City': 'Unknown',
        'Ship To State': 'Unknown',
        'Ship Zip': '00000'
    }
    
    # Only add Account Class as default if no Class-related column exists
    if 'Account Class' not in df_with_defaults.columns:
        required_columns['Account Class'] = 'Standard'
    
    for col, default_value in required_columns.items():
        if col not in df_with_defaults.columns:
            df_with_defaults[col] = default_value
            st.info(f"ℹ️ Added missing column '{col}' with default value '{default_value}'")
    
    # Clean up text columns and handle blank/dash values
    text_columns = ['Company Type', 'Client name', 'Account Class', 'WTB Account', 'Ship To City', 'Ship To State']
    for col in text_columns:
        if col in df_with_defaults.columns:
            # Convert to string
            df_with_defaults[col] = df_with_defaults[col].astype(str)
            
            # Handle various types of missing/blank values
            df_with_defaults[col] = df_with_defaults[col].replace({
                'nan': 'Unknown',
                'NaN': 'Unknown', 
                'None': 'Unknown',
                '': 'Unknown',
                ' ': 'Unknown',
                '-': 'Unknown',
                '--': 'Unknown',
                '---': 'Unknown'
            }).fillna('Unknown')
            
            # Clean up whitespace
            df_with_defaults[col] = df_with_defaults[col].str.strip()
    
    return df_with_defaults

def display_data_info(df):
    """Display information about the loaded data"""
    st.subheader("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    
    with col2:
        st.metric("Columns", len(df.columns))
    
    with col3:
        if 'Date' in df.columns:
            date_range = f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}"
            st.metric("Date Range", "")
            st.caption(date_range)
    
    with col4:
        if 'Amount' in df.columns:
            total_revenue = df['Amount'].sum()
            st.metric("Total Revenue", f"${total_revenue:,.0f}")
    
    # Show Account Class values if available
    if 'Account Class' in df.columns:
        st.subheader("🏷️ Account Class Values Found")
        class_counts = df['Account Class'].value_counts()
        st.write("Unique values in Account Class:")
        for value, count in class_counts.head(10).items():
            st.write(f"• **{value}**: {count} records")
        
        if len(class_counts) > 10:
            st.write(f"... and {len(class_counts) - 10} more unique values")
    
    # Show column information
    with st.expander("📋 Column Information"):
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Unique Values': [df[col].nunique() for col in df.columns],
            'Sample Value': [str(df[col].iloc[0]) if len(df) > 0 else 'N/A' for col in df.columns]
        })
        st.dataframe(col_info, use_container_width=True)

def create_data_upload_section():
    """Create the data upload interface"""
    st.header("📁 Load Your Financial Data")
    
    data_source = st.radio(
        "Choose your data source:",
        ["Upload File", "Local File Path", "Use Sample Data"],
        horizontal=True
    )
    
    df = None
    
    if data_source == "Upload File":
        st.subheader("📤 Upload Your Data File")
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your financial data in CSV or Excel format"
        )
        
        if uploaded_file is not None:
            df = load_data_from_file(uploaded_file=uploaded_file)
    
    elif data_source == "Local File Path":
        st.subheader("📂 Local File Path")
        file_path = st.text_input(
            "Enter the full path to your data file:",
            placeholder="C:/Users/YourName/Documents/data.csv",
            help="Enter the complete file path including filename and extension"
        )
        
        if file_path and st.button("Load Data"):
            df = load_data_from_file(file_path=file_path)
    
    else:  # Use Sample Data
        st.subheader("🎲 Sample Data")
        st.info("Generate sample data for testing the dashboard")
        
        if st.button("Generate Sample Data"):
            with st.spinner("Generating sample data..."):
                df = load_sample_data()
            st.success(f"✅ Generated {len(df)} sample records")
    
    return df

def create_column_mapping_interface(df):
    """Create interface for manual column mapping"""
    if df is None:
        return df
    
    st.header("🔄 Column Mapping (Optional)")
    
    with st.expander("Map your columns to dashboard fields"):
        st.write("If your column names don't match exactly, map them here:")
        
        required_fields = {
            'Date': 'Date/Time column',
            'Amount': 'Revenue/Amount column',
            'Client name': 'Customer/Client column'
        }
        
        col1, col2 = st.columns(2)
        column_mapping = {}
        
        for i, (field, description) in enumerate(required_fields.items()):
            with col1 if i % 2 == 0 else col2:
                selected_col = st.selectbox(
                    f"{field} ({description})",
                    options=[''] + list(df.columns),
                    key=f"mapping_{field}"
                )
                if selected_col:
                    column_mapping[selected_col] = field
        
        if column_mapping and st.button("Apply Column Mapping"):
            df_mapped = df.rename(columns=column_mapping)
            st.success("✅ Column mapping applied successfully!")
            return df_mapped
    
    return df

def load_sample_data():
    """Generate sample financial data"""
    np.random.seed(42)
    
    company_types = ['Enterprise', 'SMB', 'Mid-Market', 'Startup']
    account_classes = ['Premium', 'Standard', 'Basic', 'Enterprise']
    wtb_accounts = ['Sales', 'Marketing', 'Operations', 'Corporate', 'Regional']
    clients = [f"Client_{i:03d}" for i in range(1, 51)]
    
    locations = [
        ('New York', 'NY', '10001'), ('Los Angeles', 'CA', '90210'),
        ('Chicago', 'IL', '60601'), ('Houston', 'TX', '77001'),
        ('Phoenix', 'AZ', '85001'), ('Philadelphia', 'PA', '19101')
    ]
    
    start_date = datetime.now() - timedelta(days=730)
    dates = pd.date_range(start=start_date, periods=730, freq='D')
    
    records = []
    for _ in range(1000):
        date = pd.to_datetime(np.random.choice(dates))
        city, state, zip_code = locations[np.random.randint(0, len(locations))]
        
        record = {
            'Company Type': np.random.choice(company_types),
            'Date': date,
            'Client name': np.random.choice(clients),
            'Account Class': np.random.choice(account_classes),
            'Amount': np.random.exponential(scale=5000) + 100,
            'Month': date.strftime('%Y-%m'),
            'WTB Account': np.random.choice(wtb_accounts),
            'Ship To City': city,
            'Ship To State': state,
            'Ship Zip': zip_code
        }
        records.append(record)
    
    return pd.DataFrame(records)

def calculate_metrics(df):
    """Calculate key metrics including profitability by entity"""
    metrics = {}
    
    # Basic revenue metrics
    metrics['total_revenue'] = df['Amount'].sum()
    metrics['average_deal_size'] = df['Amount'].mean()
    metrics['total_transactions'] = len(df)
    
    # Entity-level Analysis (if Company column is available)
    if 'Company' in df.columns:
        # Entity breakdown
        entity_summary = df.groupby('Company')['Amount'].agg(['sum', 'count', 'mean']).round(2)
        entity_summary.columns = ['Total Revenue', 'Transactions', 'Avg Deal Size']
        metrics['entity_breakdown'] = entity_summary.sort_values('Total Revenue', ascending=False)
        
        # Entity count
        metrics['total_entities'] = df['Company'].nunique()
    else:
        metrics['entity_breakdown'] = pd.DataFrame()
        metrics['total_entities'] = 1
    
    # Profitability Analysis (if Account mapping is available)
    if 'Account mapping' in df.columns:
        # Convert to string and standardize
        df['Account mapping'] = df['Account mapping'].astype(str).str.upper().str.strip()
        
        # Calculate Sales vs COGS - Present sales as positive
        sales_mask = df['Account mapping'].str.contains('SALES|REVENUE|INCOME', case=False, na=False)
        cogs_mask = df['Account mapping'].str.contains('COGS|COST|EXPENSE', case=False, na=False)
        
        # Get absolute values for sales (show as positive) but keep COGS calculation logic
        sales_data = df[sales_mask]['Amount']
        sales_amount = abs(sales_data.sum())  # Present as positive
        
        cogs_data = df[cogs_mask]['Amount']
        cogs_amount = abs(cogs_data.sum())  # COGS should also be positive for display
        
        metrics['total_sales'] = sales_amount
        metrics['total_cogs'] = cogs_amount
        metrics['gross_profit'] = sales_amount - cogs_amount  # Sales - COGS
        
        if sales_amount > 0:
            metrics['gross_margin_percent'] = (metrics['gross_profit'] / sales_amount) * 100
        else:
            metrics['gross_margin_percent'] = 0
            
        # Account mapping breakdown - show absolute values for presentation
        mapping_summary = df.groupby('Account mapping')['Amount'].sum()
        # Convert to absolute values for display while preserving the account type
        mapping_summary_display = mapping_summary.abs().sort_values(ascending=False)
        metrics['account_mapping_breakdown'] = mapping_summary_display
        
        # Detailed breakdown showing sales vs cogs classification
        account_breakdown_detail = []
        for account_type, amount in mapping_summary.items():
            is_sales = any(term in account_type.upper() for term in ['SALES', 'REVENUE', 'INCOME'])
            is_cogs = any(term in account_type.upper() for term in ['COGS', 'COST', 'EXPENSE'])
            
            category = 'Sales' if is_sales else ('COGS' if is_cogs else 'Other')
            
            account_breakdown_detail.append({
                'Account Type': account_type,
                'Category': category,
                'Amount': abs(amount),  # Show as positive
                'Original Amount': amount  # Keep original for calculations
            })
        
        metrics['account_breakdown_detail'] = pd.DataFrame(account_breakdown_detail)
        
        # Entity-level profitability (if both Company and Account mapping exist)
        if 'Company' in df.columns:
            entity_profitability = []
            
            for company in df['Company'].unique():
                company_data = df[df['Company'] == company]
                
                company_sales_mask = company_data['Account mapping'].str.contains('SALES|REVENUE|INCOME', case=False, na=False)
                company_cogs_mask = company_data['Account mapping'].str.contains('COGS|COST|EXPENSE', case=False, na=False)
                
                # Present sales as positive, handle COGS properly
                company_sales = abs(company_data[company_sales_mask]['Amount'].sum())
                company_cogs = abs(company_data[company_cogs_mask]['Amount'].sum())
                company_profit = company_sales - company_cogs
                company_margin = (company_profit / company_sales * 100) if company_sales > 0 else 0
                
                # Only include entities with actual sales or COGS data
                if company_sales > 0 or company_cogs > 0:
                    entity_profitability.append({
                        'Company': company,
                        'Sales': company_sales,
                        'COGS': company_cogs,
                        'Gross Profit': company_profit,
                        'Gross Margin %': company_margin,
                        'Total Revenue': abs(company_data['Amount'].sum()),
                        'Transactions': len(company_data)
                    })
            
            metrics['entity_profitability'] = pd.DataFrame(entity_profitability).sort_values('Gross Profit', ascending=False)
        else:
            metrics['entity_profitability'] = pd.DataFrame()
            
    else:
        metrics['total_sales'] = metrics['total_revenue']
        metrics['total_cogs'] = 0
        metrics['gross_profit'] = metrics['total_revenue']
        metrics['gross_margin_percent'] = 100
        metrics['account_mapping_breakdown'] = pd.Series()
        metrics['entity_profitability'] = pd.DataFrame()
        metrics['account_breakdown_detail'] = pd.DataFrame()
    
    # Month over month growth
    if 'Month' in df.columns:
        df_sorted = df.sort_values('Date')
        current_month = df_sorted['Month'].iloc[-1]
        previous_month = pd.to_datetime(current_month) - pd.DateOffset(months=1)
        previous_month_str = previous_month.strftime('%Y-%m')
        
        current_month_revenue = df[df['Month'] == current_month]['Amount'].sum()
        previous_month_revenue = df[df['Month'] == previous_month_str]['Amount'].sum()
        
        if previous_month_revenue > 0:
            metrics['mom_growth'] = ((current_month_revenue - previous_month_revenue) / previous_month_revenue) * 100
        else:
            metrics['mom_growth'] = 0
    else:
        metrics['mom_growth'] = 0
    
    return metrics

def create_revenue_trend_chart(df, breakdown_by='Company Type'):
    """Create revenue trend chart"""
    df_grouped = df.groupby(['Date', breakdown_by])['Amount'].sum().reset_index()
    
    fig = px.line(
        df_grouped, 
        x='Date', 
        y='Amount', 
        color=breakdown_by,
        title=f'Revenue Trends by {breakdown_by}'
    )
    
    fig.update_layout(height=500)
    return fig

def create_pivot_table_interface(df):
    """Create interactive pivot table functionality"""
    st.header("📊 Pivot Table Analysis")
    
    # Get available columns for pivot analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rows = st.multiselect(
            "Rows (Group by):",
            options=categorical_cols,
            default=['Account Class'] if 'Account Class' in categorical_cols else categorical_cols[:1]
        )
    
    with col2:
        columns = st.multiselect(
            "Columns (Pivot by):",
            options=categorical_cols,
            default=['Account mapping'] if 'Account mapping' in categorical_cols else []
        )
    
    with col3:
        values = st.selectbox(
            "Values (Aggregate):",
            options=numeric_cols,
            index=numeric_cols.index('Amount') if 'Amount' in numeric_cols else 0
        )
    
    aggregation = st.selectbox(
        "Aggregation Function:",
        options=['sum', 'mean', 'count', 'min', 'max'],
        index=0
    )
    
    if rows or columns:
        try:
            if columns:
                # Create pivot table with both rows and columns
                pivot = pd.pivot_table(
                    df, 
                    values=values, 
                    index=rows if rows else None, 
                    columns=columns, 
                    aggfunc=aggregation, 
                    fill_value=0,
                    margins=True,
                    margins_name='Total'
                )
            else:
                # Create simple groupby if no columns specified
                pivot = df.groupby(rows)[values].agg(aggregation).reset_index()
            
            st.subheader("📈 Pivot Table Results")
            
            # Format currency if it's Amount column
            if values == 'Amount':
                if isinstance(pivot, pd.DataFrame) and len(pivot.columns) > 1:
                    st.dataframe(pivot.style.format('${:,.2f}'), use_container_width=True)
                else:
                    st.dataframe(pivot.style.format({values: '${:,.2f}'}), use_container_width=True)
            else:
                st.dataframe(pivot, use_container_width=True)
            
            # Create visualization of pivot results
            if len(rows) == 1 and not columns and len(pivot) <= 20:
                fig = px.bar(
                    x=pivot.index if isinstance(pivot.index, pd.Index) else pivot[rows[0]],
                    y=pivot[values] if values in pivot.columns else pivot.values,
                    title=f"{aggregation.title()} of {values} by {rows[0]}",
                    labels={'x': rows[0], 'y': f"{aggregation.title()} {values}"}
                )
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error creating pivot table: {str(e)}")
            st.info("Try selecting different columns or check your data format.")
    
    return None

def main():
    # Dashboard header
    st.title("🏢 Financial Analysis Dashboard")
    st.markdown("### Comprehensive Revenue Analytics & Business Intelligence Platform")
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Data loading section
    if not st.session_state.data_loaded:
        df = create_data_upload_section()
        
        if df is not None:
            # Optional column mapping
            df = create_column_mapping_interface(df)
            
            if st.button("✅ Use This Data"):
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.success("✅ Data loaded! Dashboard ready...")
                st.rerun()
        else:
            st.info("👆 Please load your data using one of the options above to continue.")
            return
    
    # Add change data source button in sidebar
    if st.sidebar.button("🔄 Change Data Source"):
        st.session_state.data_loaded = False
        st.rerun()
    
    # Use stored data (with safety check)
    if 'df' not in st.session_state:
        st.session_state.data_loaded = False
        st.rerun()
        return
    
    df = st.session_state.df
    
    # Validate required columns
    required_cols = ['Date', 'Amount']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
        st.info("Please use the column mapping feature or ensure your data has Date and Amount columns.")
        if st.button("🔄 Go Back to Data Loading"):
            st.session_state.data_loaded = False
            st.rerun()
        return
    
    # Sidebar filters
    st.sidebar.header("📊 Dashboard Controls")
    
    # Data info
    st.sidebar.metric("Total Records", f"{len(df):,}")
    
    # Date range filter
    if 'Date' in df.columns:
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(df['Date'].min().date(), df['Date'].max().date()),
            min_value=df['Date'].min().date(),
            max_value=df['Date'].max().date()
        )
        
        # Apply date filter
        if len(date_range) == 2:
            start_date, end_date = date_range
            df_filtered = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]
        else:
            df_filtered = df
    else:
        df_filtered = df
    
    # Company/Entity filter (if available)
    if 'Company' in df.columns and len(df['Company'].unique()) > 1:
        # Convert to string and handle NaN values
        df['Company'] = df['Company'].astype(str).fillna('Unknown')
        company_options = sorted(df['Company'].unique())
        selected_companies = st.sidebar.multiselect(
            "Select Companies/Entities",
            options=company_options,
            default=company_options
        )
        
        if selected_companies:
            df_filtered = df_filtered[df_filtered['Company'].isin(selected_companies)]
    
    # Account Mapping filter (if available) 
    if 'Account mapping' in df.columns and len(df['Account mapping'].unique()) > 1:
        df['Account mapping'] = df['Account mapping'].astype(str).fillna('Unknown')
        mapping_options = sorted(df['Account mapping'].unique())
        selected_mappings = st.sidebar.multiselect(
            "Select Account Mappings",
            options=mapping_options,
            default=mapping_options
        )
        
        if selected_mappings:
            df_filtered = df_filtered[df_filtered['Account mapping'].isin(selected_mappings)]
    
    # Client filter (if available)
    if 'Client name' in df.columns and len(df['Client name'].unique()) > 1:
        # Convert to string and handle NaN values
        df['Client name'] = df['Client name'].astype(str).fillna('Unknown')
        client_options = sorted(df['Client name'].unique())
        selected_clients = st.sidebar.multiselect(
            "Select Clients",
            options=client_options,
            default=client_options[:min(10, len(client_options))]  # Default to top 10 or all if less
        )
        
        if selected_clients:
            df_filtered = df_filtered[df_filtered['Client name'].isin(selected_clients)]
    
    # Account Class filter (if available)
    if 'Account Class' in df.columns and len(df['Account Class'].unique()) > 1:
        # Convert to string and handle NaN values
        df['Account Class'] = df['Account Class'].astype(str).fillna('Unknown')
        account_options = sorted(df['Account Class'].unique())
        selected_accounts = st.sidebar.multiselect(
            "Select Account Classes",
            options=account_options,
            default=account_options
        )
        
        if selected_accounts:
            df_filtered = df_filtered[df_filtered['Account Class'].isin(selected_accounts)]
    
    # Check if data exists after filtering
    if df_filtered.empty:
        st.error("No data available with current filters. Please adjust your selections.")
        return
    
    # Calculate metrics
    metrics = calculate_metrics(df_filtered)
    
    # Display KPI cards - Enhanced with entity and profitability
    if 'Company' in df.columns and len(metrics['entity_breakdown']) > 0:
        # Show entity summary first
        st.subheader("🏢 Entity Performance Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Entities", metrics['total_entities'])
        with col2:
            st.metric("Total Revenue", f"${metrics['total_revenue']:,.0f}")
        with col3:
            if 'Account mapping' in df.columns:
                st.metric("Gross Margin", f"{metrics['gross_margin_percent']:.1f}%")
            else:
                st.metric("Avg Deal Size", f"${metrics['average_deal_size']:,.0f}")
        
        # Entity breakdown table
        st.dataframe(metrics['entity_breakdown'].style.format({
            'Total Revenue': '${:,.2f}',
            'Avg Deal Size': '${:,.2f}'
        }), use_container_width=True)
        
        # Entity profitability (if available)
        if len(metrics['entity_profitability']) > 0:
            st.subheader("💰 Entity Profitability")
            st.dataframe(metrics['entity_profitability'].style.format({
                'Sales': '${:,.2f}',
                'COGS': '${:,.2f}',
                'Gross Profit': '${:,.2f}',
                'Gross Margin %': '{:.1f}%',
                'Total Revenue': '${:,.2f}'
            }), use_container_width=True)
    
    elif 'Account mapping' in df.columns and metrics['total_sales'] > 0:
        # Show profitability KPIs without entity breakdown
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Total Sales", f"${metrics['total_sales']:,.0f}")
        
        with col2:
            st.metric("Total COGS", f"${metrics['total_cogs']:,.0f}")
        
        with col3:
            st.metric("Gross Profit", f"${metrics['gross_profit']:,.0f}")
        
        with col4:
            st.metric("Gross Margin", f"{metrics['gross_margin_percent']:.1f}%")
        
        with col5:
            st.metric("Avg Deal Size", f"${metrics['average_deal_size']:,.0f}")
        
        with col6:
            st.metric("MoM Growth", f"{metrics['mom_growth']:+.1f}%")
            
        # Account Mapping Breakdown
        if len(metrics['account_mapping_breakdown']) > 0:
            st.subheader("💰 Account Mapping Breakdown")
            
            col1, col2 = st.columns(2)
            
            with col1:
                mapping_df = metrics['account_mapping_breakdown'].reset_index()
                mapping_df.columns = ['Account Mapping', 'Amount']
                mapping_df['Amount_Formatted'] = mapping_df['Amount'].apply(lambda x: f"${x:,.2f}")
                
                st.dataframe(mapping_df[['Account Mapping', 'Amount_Formatted']], 
                           use_container_width=True, hide_index=True)
            
            with col2:
                fig_mapping = px.pie(
                    values=metrics['account_mapping_breakdown'].values,
                    names=metrics['account_mapping_breakdown'].index,
                    title="Amount Distribution by Account Mapping"
                )
                st.plotly_chart(fig_mapping, use_container_width=True)
                
    else:
        # Standard KPIs if no Account mapping or Company
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Revenue", f"${metrics['total_revenue']:,.0f}")
        
        with col2:
            st.metric("Average Deal Size", f"${metrics['average_deal_size']:,.0f}")
        
        with col3:
            st.metric("MoM Growth", f"{metrics['mom_growth']:+.1f}%")
        
        with col4:
            st.metric("Total Transactions", f"{metrics['total_transactions']:,}")
    
    # Update breakdown options to include Company
    breakdown_options = []
    if 'Company' in df.columns and len(df['Company'].unique()) > 1:
        breakdown_options.append('Company')
    if 'Company Type' in df.columns and len(df['Company Type'].unique()) > 1:
        breakdown_options.append('Company Type')
    if 'Account Class' in df.columns and len(df['Account Class'].unique()) > 1:
        breakdown_options.append('Account Class')
    if 'Account mapping' in df.columns and len(df['Account mapping'].unique()) > 1:
        breakdown_options.append('Account mapping')
    if 'WTB Account' in df.columns and len(df['WTB Account'].unique()) > 1:
        breakdown_options.append('WTB Account')
    
    if not breakdown_options:
        breakdown_options = ['Client name']  # Fallback to client name
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Revenue Trends", 
        "💰 Profitability Analysis",
        "📊 Pivot Tables",
        "🌍 Geographic Analysis", 
        "👥 Client Performance", 
        "📋 Export Data"
    ])
    
    with tab1:
        st.header("Revenue Trend Analysis")
        
        breakdown_option = st.selectbox(
            "Breakdown by:",
            breakdown_options
        )
        
        trend_chart = create_revenue_trend_chart(df_filtered, breakdown_option)
        st.plotly_chart(trend_chart, use_container_width=True)
        
        # Revenue distribution
        if len(breakdown_options) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                if breakdown_options[0] in df_filtered.columns:
                    revenue_data = df_filtered.groupby(breakdown_options[0])['Amount'].sum()
                    fig_pie = px.pie(
                        values=revenue_data.values, 
                        names=revenue_data.index,
                        title=f"Revenue by {breakdown_options[0]}"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                if len(breakdown_options) > 1 and breakdown_options[1] in df_filtered.columns:
                    revenue_data2 = df_filtered.groupby(breakdown_options[1])['Amount'].sum()
                    fig_pie2 = px.pie(
                        values=revenue_data2.values, 
                        names=revenue_data2.index,
                        title=f"Revenue by {breakdown_options[1]}"
                    )
                    st.plotly_chart(fig_pie2, use_container_width=True)
    
    with tab2:
        st.header("💰 Profitability Analysis")
        
        if 'Account mapping' in df_filtered.columns:
            # Sales vs COGS Analysis with detailed breakdown
            st.subheader("📊 Account Mapping Breakdown")
            
            if len(metrics['account_breakdown_detail']) > 0:
                # Show detailed account mapping table
                detail_table = metrics['account_breakdown_detail'][['Account Type', 'Category', 'Amount']].copy()
                detail_table = detail_table.sort_values('Amount', ascending=False)
                
                st.dataframe(detail_table.style.format({
                    'Amount': '${:,.2f}'
                }), use_container_width=True, hide_index=True)
                
                # Summary by category
                category_summary = detail_table.groupby('Category')['Amount'].sum().reset_index()
                category_summary = category_summary.sort_values('Amount', ascending=False)
                
                st.subheader("📈 Sales vs COGS Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Summary table
                    st.dataframe(category_summary.style.format({
                        'Amount': '${:,.2f}'
                    }), use_container_width=True, hide_index=True)
                
                with col2:
                    # Pie chart
                    if len(category_summary) > 0:
                        fig_category = px.pie(
                            category_summary,
                            values='Amount',
                            names='Category',
                            title="Amount Distribution by Category"
                        )
                        st.plotly_chart(fig_category, use_container_width=True)
                
                # Profitability metrics table
                st.subheader("💹 Profitability Metrics")
                
                profit_metrics = pd.DataFrame({
                    'Metric': ['Total Sales', 'Total COGS', 'Gross Profit', 'Gross Margin %'],
                    'Value': [
                        f"${metrics['total_sales']:,.2f}",
                        f"${metrics['total_cogs']:,.2f}",
                        f"${metrics['gross_profit']:,.2f}",
                        f"{metrics['gross_margin_percent']:.1f}%"
                    ]
                })
                
                st.dataframe(profit_metrics, use_container_width=True, hide_index=True)
                
                # Monthly profitability trend (if Date available)
                if 'Date' in df_filtered.columns:
                    st.subheader("📅 Monthly Profitability Trend")
                    
                    monthly_profit = []
                    for month in sorted(df_filtered['Month'].unique()):
                        month_data = df_filtered[df_filtered['Month'] == month].copy()
                        month_data['Account mapping'] = month_data['Account mapping'].astype(str).str.upper()
                        
                        sales_mask = month_data['Account mapping'].str.contains('SALES|REVENUE|INCOME', case=False, na=False)
                        cogs_mask = month_data['Account mapping'].str.contains('COGS|COST|EXPENSE', case=False, na=False)
                        
                        monthly_sales = abs(month_data[sales_mask]['Amount'].sum())
                        monthly_cogs = abs(month_data[cogs_mask]['Amount'].sum())
                        monthly_gp = monthly_sales - monthly_cogs
                        monthly_margin = (monthly_gp / monthly_sales * 100) if monthly_sales > 0 else 0
                        
                        monthly_profit.append({
                            'Month': month,
                            'Sales': monthly_sales,
                            'COGS': monthly_cogs,
                            'Gross Profit': monthly_gp,
                            'Gross Margin %': monthly_margin
                        })
                    
                    if monthly_profit:
                        monthly_df = pd.DataFrame(monthly_profit)
                        
                        # Show monthly table
                        st.dataframe(monthly_df.style.format({
                            'Sales': '${:,.2f}',
                            'COGS': '${:,.2f}',
                            'Gross Profit': '${:,.2f}',
                            'Gross Margin %': '{:.1f}%'
                        }), use_container_width=True, hide_index=True)
                        
                        # Gross margin trend chart
                        fig_margin_trend = px.line(
                            monthly_df,
                            x='Month',
                            y='Gross Margin %',
                            title="Monthly Gross Margin Trend",
                            markers=True
                        )
                        fig_margin_trend.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig_margin_trend, use_container_width=True)
            
            else:
                st.warning("⚠️ No account mapping data found or unable to categorize as Sales/COGS")
                st.info("Make sure your Account mapping column contains terms like 'SALES', 'REVENUE', 'COGS', 'COST', or 'EXPENSE'")
                
                # Show raw account mapping values for debugging
                st.subheader("🔍 Raw Account Mapping Values")
                raw_mappings = df_filtered['Account mapping'].value_counts().reset_index()
                raw_mappings.columns = ['Account Mapping', 'Count']
                st.dataframe(raw_mappings, use_container_width=True, hide_index=True)
            
        else:
            st.info("Account mapping column not found. Upload data with 'Account mapping' column for profitability analysis.")
    
    with tab3:
        # Pivot table functionality
        create_pivot_table_interface(df_filtered)
    
    with tab4:
        st.header("Geographic Revenue Analysis")
        
        # Check if geographic data is available
        if 'Ship To State' in df_filtered.columns and 'Ship To City' in df_filtered.columns:
            # Create geographic chart
            geo_data = df_filtered.groupby(['Ship To State', 'Ship To City'])['Amount'].sum().reset_index()
            
            fig_geo = px.treemap(
                geo_data,
                path=['Ship To State', 'Ship To City'],
                values='Amount',
                title='Revenue Distribution by Geography'
            )
            fig_geo.update_layout(height=600)
            st.plotly_chart(fig_geo, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top States by Revenue")
                state_revenue = df_filtered.groupby('Ship To State')['Amount'].sum().sort_values(ascending=False)
                if len(state_revenue) > 0:
                    fig_states = px.bar(
                        x=state_revenue.index[:10], 
                        y=state_revenue.values[:10],
                        title="Top 10 States by Revenue"
                    )
                    st.plotly_chart(fig_states, use_container_width=True)
            
            with col2:
                st.subheader("Top Cities by Revenue")
                city_revenue = df_filtered.groupby('Ship To City')['Amount'].sum().sort_values(ascending=False)
                if len(city_revenue) > 0:
                    fig_cities = px.bar(
                        x=city_revenue.index[:10], 
                        y=city_revenue.values[:10],
                        title="Top 10 Cities by Revenue"
                    )
                    fig_cities.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_cities, use_container_width=True)
        else:
            st.info("Geographic data not available. Add 'Ship To State' and 'Ship To City' columns for geographic analysis.")
    with tab5:
        st.header("Client Performance Analysis")
        
        if 'Client name' in df_filtered.columns:
            # Top clients
            top_clients = df_filtered.groupby('Client name')['Amount'].agg(['sum', 'count', 'mean']).sort_values('sum', ascending=False).head(10)
            top_clients.columns = ['Total Revenue', 'Transactions', 'Avg Deal Size']
            
            st.subheader("Top 10 Clients by Revenue")
            st.dataframe(top_clients.style.format({
                'Total Revenue': '${:,.2f}',
                'Avg Deal Size': '${:,.2f}'
            }))
            
            # Client profitability (if Account mapping available)
            if 'Account mapping' in df_filtered.columns:
                st.subheader("Client Profitability Analysis")
                
                client_profit = df_filtered.copy()
                client_profit['Account mapping'] = client_profit['Account mapping'].astype(str).str.upper()
                
                # Categorize sales vs cogs by client
                sales_mask = client_profit['Account mapping'].str.contains('SALES|REVENUE|INCOME', case=False, na=False)
                cogs_mask = client_profit['Account mapping'].str.contains('COGS|COST|EXPENSE', case=False, na=False)
                
                # Calculate client-level profitability with absolute values
                client_profitability = []
                
                for client in client_profit['Client name'].unique():
                    client_data = client_profit[client_profit['Client name'] == client]
                    
                    client_sales = abs(client_data[sales_mask]['Amount'].sum()) if sales_mask.any() else 0
                    client_cogs = abs(client_data[cogs_mask]['Amount'].sum()) if cogs_mask.any() else 0
                    client_gp = client_sales - client_cogs
                    client_margin = (client_gp / client_sales * 100) if client_sales > 0 else 0
                    client_total = abs(client_data['Amount'].sum())
                    
                    # Only include clients with actual sales or COGS activity
                    if client_sales > 0 or client_cogs > 0:
                        client_profitability.append({
                            'Client': client,
                            'Sales': client_sales,
                            'COGS': client_cogs,
                            'Gross Profit': client_gp,
                            'Gross Margin %': client_margin,
                            'Total Activity': client_total,
                            'Transactions': len(client_data)
                        })
                
                if client_profitability:
                    client_profit_df = pd.DataFrame(client_profitability)
                    client_profit_df = client_profit_df.sort_values('Gross Profit', ascending=False).head(15)
                    
                    st.dataframe(client_profit_df.style.format({
                        'Sales': '${:,.2f}',
                        'COGS': '${:,.2f}',
                        'Gross Profit': '${:,.2f}',
                        'Gross Margin %': '{:.1f}%',
                        'Total Activity': '${:,.2f}'
                    }), use_container_width=True, hide_index=True)
                    
                    # Summary stats
                    st.subheader("📊 Client Profitability Summary")
                    
                    summary_stats = pd.DataFrame({
                        'Metric': [
                            'Clients with Sales Activity',
                            'Clients with COGS Activity', 
                            'Avg Client Gross Margin',
                            'Most Profitable Client',
                            'Highest Margin Client'
                        ],
                        'Value': [
                            f"{len(client_profit_df[client_profit_df['Sales'] > 0])}",
                            f"{len(client_profit_df[client_profit_df['COGS'] > 0])}",
                            f"{client_profit_df['Gross Margin %'].mean():.1f}%",
                            f"{client_profit_df.iloc[0]['Client']} (${client_profit_df.iloc[0]['Gross Profit']:,.0f})" if len(client_profit_df) > 0 else "N/A",
                            f"{client_profit_df.loc[client_profit_df['Gross Margin %'].idxmax(), 'Client']} ({client_profit_df['Gross Margin %'].max():.1f}%)" if len(client_profit_df) > 0 else "N/A"
                        ]
                    })
                    
                    st.dataframe(summary_stats, use_container_width=True, hide_index=True)
                    
                else:
                    st.warning("⚠️ No clients found with identifiable Sales or COGS activity")
                    st.info("This could mean:")
                    st.write("• Account mapping values don't contain sales/revenue/income or cogs/cost/expense terms")
                    st.write("• All transactions are categorized as 'Other'")
                    st.write("• Data format needs adjustment")
            else:
                st.info("Account mapping data not available for client profitability analysis.")
            
            # Client revenue distribution
            client_revenue_dist = df_filtered.groupby('Client name')['Amount'].sum()
            fig_hist = px.histogram(
                x=client_revenue_dist.values,
                nbins=min(20, len(client_revenue_dist)),
                title="Distribution of Client Revenue"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Client data not available. Add 'Client name' column for client analysis.")
    
    with tab6:
        st.header("Export & Reporting")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📥 Data Export")
            
            if st.button("Export Filtered Data to CSV"):
                csv_data = df_filtered.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"financial_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            # Export profitability summary
            if 'Account mapping' in df_filtered.columns:
                if st.button("Export Profitability Summary"):
                    profit_summary = create_profitability_summary(df_filtered)
                    csv_profit = profit_summary.to_csv(index=True)
                    st.download_button(
                        label="Download Profitability CSV",
                        data=csv_profit,
                        file_name=f"profitability_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
        
        with col2:
            st.subheader("📊 Executive Summary")
            
            summary = f"""
            **Financial Performance Summary**
            
            📈 **Key Metrics:**
            - Total Revenue: ${metrics['total_revenue']:,.2f}
            - Average Deal Size: ${metrics['average_deal_size']:,.2f}
            - Total Transactions: {metrics['total_transactions']:,}
            - Month-over-Month Growth: {metrics['mom_growth']:+.1f}%
            """
            
            if 'Account mapping' in df_filtered.columns and metrics['total_sales'] > 0:
                summary += f"""
            
            💰 **Profitability:**
            - Total Sales: ${metrics['total_sales']:,.2f}
            - Total COGS: ${metrics['total_cogs']:,.2f}
            - Gross Profit: ${metrics['gross_profit']:,.2f}
            - Gross Margin: {metrics['gross_margin_percent']:.1f}%
                """
            
            summary += f"""
            
            📊 **Data Overview:**
            - Records Analyzed: {len(df_filtered):,}
            - Original Records: {len(df):,}
            """
            
            if 'Date' in df.columns:
                summary += f"- Date Range: {df_filtered['Date'].min().strftime('%Y-%m-%d')} to {df_filtered['Date'].max().strftime('%Y-%m-%d')}"
            
            st.markdown(summary)

def create_profitability_summary(df):
    """Create profitability summary for export"""
    if 'Account mapping' not in df.columns:
        return pd.DataFrame()
    
    df_profit = df.copy()
    df_profit['Account mapping'] = df_profit['Account mapping'].astype(str).str.upper()
    
    # Monthly profitability
    monthly_summary = []
    for month in df_profit['Month'].unique():
        month_data = df_profit[df_profit['Month'] == month]
        
        sales_mask = month_data['Account mapping'].str.contains('SALES|REVENUE|INCOME', case=False, na=False)
        cogs_mask = month_data['Account mapping'].str.contains('COGS|COST|EXPENSE', case=False, na=False)
        
        sales = month_data[sales_mask]['Amount'].sum()
        cogs = month_data[cogs_mask]['Amount'].sum()
        
        monthly_summary.append({
            'Month': month,
            'Sales': sales,
            'COGS': cogs,
            'Gross_Profit': sales - cogs,
            'Gross_Margin_Percent': (sales - cogs) / sales * 100 if sales > 0 else 0
        })
    
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Client data not available. Add 'Client name' column for client analysis.")
    
    with tab6:
        st.header("Export & Reporting")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📥 Data Export")
            
            if st.button("Export Filtered Data to CSV"):
                csv_data = df_filtered.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"financial_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            st.subheader("📊 Summary Report")
            
            summary = f"""
            **Financial Performance Summary**
            
            📈 **Key Metrics:**
            - Total Revenue: ${metrics['total_revenue']:,.2f}
            - Average Deal Size: ${metrics['average_deal_size']:,.2f}
            - Total Transactions: {metrics['total_transactions']:,}
            - Month-over-Month Growth: {metrics['mom_growth']:+.1f}%
            
            📊 **Data Overview:**
            - Records Analyzed: {len(df_filtered):,}
            - Original Records: {len(df):,}
            """
            
            if 'Date' in df.columns:
                summary += f"- Date Range: {df_filtered['Date'].min().strftime('%Y-%m-%d')} to {df_filtered['Date'].max().strftime('%Y-%m-%d')}"
            
            st.markdown(summary)

if __name__ == "__main__":
    main()