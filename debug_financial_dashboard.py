import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io

# Page configuration
st.set_page_config(
    page_title="Financial Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    """Calculate key metrics"""
    metrics = {}
    metrics['total_revenue'] = df['Amount'].sum()
    metrics['average_deal_size'] = df['Amount'].mean()
    metrics['total_transactions'] = len(df)
    
    # Month over month growth
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

def create_geographic_chart(df):
    """Create geographic revenue chart"""
    geo_data = df.groupby(['Ship To State', 'Ship To City'])['Amount'].sum().reset_index()
    
    fig = px.treemap(
        geo_data,
        path=['Ship To State', 'Ship To City'],
        values='Amount',
        title='Revenue Distribution by Geography'
    )
    
    fig.update_layout(height=600)
    return fig

def main():
    # Dashboard header
    st.title("🏢 Financial Analysis Dashboard")
    st.markdown("### Comprehensive Revenue Analytics & Business Intelligence Platform")
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Data loading section
    if not st.session_state.data_loaded:
        st.header("📁 Data Source Configuration")
        
        if st.button("Generate Sample Data"):
            with st.spinner("Generating sample data..."):
                df = load_sample_data()
            st.success(f"✅ Generated {len(df)} sample records")
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.rerun()
        else:
            st.info("Click 'Generate Sample Data' to get started")
            return
    
    # Use stored data
    df = st.session_state.df
    
    # Sidebar filters
    st.sidebar.header("📊 Dashboard Controls")
    
    if st.sidebar.button("🔄 Change Data Source"):
        st.session_state.data_loaded = False
        st.rerun()
    
    # Date range filter
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
    
    # Additional filters
    selected_clients = st.sidebar.multiselect(
        "Select Clients",
        options=sorted(df['Client name'].unique()),
        default=sorted(df['Client name'].unique())[:10]
    )
    
    selected_account_classes = st.sidebar.multiselect(
        "Select Account Classes",
        options=sorted(df['Account Class'].unique()),
        default=sorted(df['Account Class'].unique())
    )
    
    # Apply filters
    df_filtered = df_filtered[
        (df_filtered['Client name'].isin(selected_clients)) &
        (df_filtered['Account Class'].isin(selected_account_classes))
    ]
    
    # Check if data exists after filtering
    if df_filtered.empty:
        st.error("No data available with current filters. Please adjust your selections.")
        return
    
    # Calculate metrics
    metrics = calculate_metrics(df_filtered)
    
    # Display KPI cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Revenue", f"${metrics['total_revenue']:,.0f}")
    
    with col2:
        st.metric("Average Deal Size", f"${metrics['average_deal_size']:,.0f}")
    
    with col3:
        st.metric("MoM Growth", f"{metrics['mom_growth']:+.1f}%")
    
    with col4:
        st.metric("Total Transactions", f"{metrics['total_transactions']:,}")
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Revenue Trends", 
        "🌍 Geographic Analysis", 
        "👥 Client Performance", 
        "📋 Export Data"
    ])
    
    with tab1:
        st.header("Revenue Trend Analysis")
        
        breakdown_option = st.selectbox(
            "Breakdown by:",
            ['Company Type', 'Account Class', 'WTB Account']
        )
        
        trend_chart = create_revenue_trend_chart(df_filtered, breakdown_option)
        st.plotly_chart(trend_chart, use_container_width=True)
        
        # Revenue distribution
        col1, col2 = st.columns(2)
        
        with col1:
            company_revenue = df_filtered.groupby('Company Type')['Amount'].sum()
            fig_pie = px.pie(
                values=company_revenue.values, 
                names=company_revenue.index,
                title="Revenue by Company Type"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            account_revenue = df_filtered.groupby('Account Class')['Amount'].sum()
            fig_pie2 = px.pie(
                values=account_revenue.values, 
                names=account_revenue.index,
                title="Revenue by Account Class"
            )
            st.plotly_chart(fig_pie2, use_container_width=True)
    
    with tab2:
        st.header("Geographic Revenue Analysis")
        
        geo_chart = create_geographic_chart(df_filtered)
        st.plotly_chart(geo_chart, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top States by Revenue")
            state_revenue = df_filtered.groupby('Ship To State')['Amount'].sum().sort_values(ascending=False)
            fig_states = px.bar(
                x=state_revenue.index, 
                y=state_revenue.values,
                title="Revenue by State"
            )
            st.plotly_chart(fig_states, use_container_width=True)
        
        with col2:
            st.subheader("Top Cities by Revenue")
            city_revenue = df_filtered.groupby('Ship To City')['Amount'].sum().sort_values(ascending=False)
            fig_cities = px.bar(
                x=city_revenue.index, 
                y=city_revenue.values,
                title="Revenue by City"
            )
            fig_cities.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_cities, use_container_width=True)
    
    with tab3:
        st.header("Client Performance Analysis")
        
        # Top clients
        top_clients = df_filtered.groupby('Client name')['Amount'].agg(['sum', 'count', 'mean']).sort_values('sum', ascending=False).head(10)
        top_clients.columns = ['Total Revenue', 'Transactions', 'Avg Deal Size']
        
        st.subheader("Top 10 Clients by Revenue")
        st.dataframe(top_clients.style.format({
            'Total Revenue': '${:,.2f}',
            'Avg Deal Size': '${:,.2f}'
        }))
        
        # Client revenue distribution
        client_revenue_dist = df_filtered.groupby('Client name')['Amount'].sum()
        fig_hist = px.histogram(
            x=client_revenue_dist.values,
            nbins=20,
            title="Distribution of Client Revenue"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with tab4:
        st.header("Export & Reporting")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📥 Data Export")
            
            if st.button("Export to CSV"):
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
            - Date Range: {df_filtered['Date'].min().strftime('%Y-%m-%d')} to {df_filtered['Date'].max().strftime('%Y-%m-%d')}
            """
            
            st.markdown(summary)

if __name__ == "__main__":
    main()