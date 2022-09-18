import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import math
import altair as alt
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="SA Equity Markets", page_icon=":bar_chart:", layout="wide", initial_sidebar_state="collapsed")
# ---- MAINPAGE ----
st.title(":bar_chart: SA Equity Dashboard")
st.markdown("""---""")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Indices Summary", "Sector Summary", "Capped SWIX", "FTSE/JSE Indices", "Valuations"])

#Data Loads
@st.cache
def get_data():
    xl = pd.ExcelFile("alldata.xlsx")
    df = pd.read_excel(xl, sheet_name='Prices', index_col=0)
    prices_df = pd.read_excel(xl, sheet_name='SWIX', parse_dates=['CLOSE'], index_col=0)
    stocks_df = pd.read_excel(xl, sheet_name='JSE')
    benchmark = pd.read_excel(xl, sheet_name='SWIX Holdings', index_col=0)
    return df, prices_df, stocks_df, benchmark

df, prices_df, stocks_df, benchmark = get_data()

#Set Up Codes DataFrame
codes = df[0:1].transpose().reset_index()
codes.columns = ['Index', 'Code']

#Clean df DataFrame
header = df.iloc[0]
df = df[1:]
df.columns = header
df.index = pd.to_datetime(df.index)
df = df.ffill(axis=0)
df = df.select_dtypes(exclude=['object'])
df = df.drop(['EFPE'], axis=1)

#Set Up Index Returns DataFrame
index_returns = df.pct_change()

#Clean Up prices_df DataFrame
prices_df = prices_df.ffill(axis=0)
prices_df = prices_df / 100

#Set Up Stock Returns DataFrame
stock_returns = prices_df.pct_change()

#Clean Up benchmark DataFrame
benchmark = benchmark.drop(['Name', 'Size', 'Industry', 'Supersector', 'Sector', 'Subsector', 'Industry Code', 'Supersector Code', 'Sector Code', 'Subsector Code'], axis=1)
benchmark = benchmark.drop('Total:')
benchmark = benchmark.loc[(benchmark != 0).any(axis=1)]
benchmark = benchmark.sort_index()
benchmark.index.name = 'Code'

#Index Lists
equity_indices = ['J203T', 'J303T', 'J403T', 'J433T', 'J200T', 'J201T', 'J202T', 'J204T', 'JI0030T', 'J257T', 'J258T', 'J330T', 'J331T']
index_sectors = ['J433T', 'JI0030T', 'J257T', 'J258T']
industry_sectors = ['JI0010T', 'JI0015T', 'JI0020T', 'JI0030T', 'JI0035T', 'JI0040T', 'JI0045T', 'JI0050T', 'JI0055T', 'JI0060T']
subsectors = ['JS1512T', 'JS2011T', 'JS2013T', 'JS3011T', 'JS3031T', 'JS4021T', 'JS4024T', 'JS4041T', 'JS4051T', 'JS4511T', 'JS4512T', 'JS4513T', 'JS4521T', 'JS5011T', 'JS5023T', 'JS5512T', 'JS5513T', 'JS5521T', 'JS6011T', 'J803T']

#Index Constituents
top40 = ['ABG', 'AGL', 'AMS', 'ANG', 'ANH', 'APN', 'BHG', 'BID', 'BTI', 'BVT', 'CFR', 'CLS', 'CPI', 'DSY', 'EXX', 'FSR', 'GFI', 'GLN', 'GRT', 'IMP', 'INL', 'INP', 'MCG', 'MNP', 'MRP', 'MTN', 'NED', 'NPH', 'NPN', 'NRP', 'OMU', 'PRX', 'REM', 'RNI', 'SBK', 'SHP', 'SLM', 'SOL', 'SSW', 'VOD', 'WHL']
midcap = ['APN', 'ARI', 'AVI', 'BAW', 'BVT', 'BYI', 'CCO', 'CLS', 'CML', 'DCP', 'DGH', 'EXX', 'FFA', 'FFB', 'GRT', 'HAR', 'HMN', 'INL', 'INP', 'ITE', 'LHC', 'MCG', 'MEI', 'MKR', 'MRP', 'MTM', 'N91', 'NED', 'NRP', 'NTC', 'NY1', 'OMU', 'PIK', 'PPH', 'PSG', 'QLT', 'RBP', 'RDF', 'REM', 'RES', 'RMI', 'RNI', 'SAP', 'SNT', 'SPP', 'SRE', 'TBS', 'TCP', 'TFG', 'TKG', 'TRU', 'TXT', 'WHL']
smallcap = ['ACL', 'ADH', 'AEL', 'AFE', 'AFH', 'AFT', 'AIL', 'AIP', 'ARL', 'ATT', 'BAT', 'BLU', 'CLH', 'COH', 'CSB', 'DRD', 'DTC', 'EMI', 'EQU', 'FBR', 'FTB', 'GND', 'HCI', 'HDC', 'HYP', 'IPF', 'JSE', 'KAP', 'KRO', 'KST', 'L2D', 'LBR', 'LTE', 'MLI', 'MSM', 'MSP', 'MTA', 'MTH', 'MUR', 'OCE', 'OMN', 'PAN', 'PPC', 'RBX', 'RFG', 'RLO', 'SAC', 'SNH', 'SPG', 'SSS', 'SSU', 'SUI', 'TGA', 'THA', 'TSG', 'VKE', 'WBO', 'ZED']
resi10 = ['AGL', 'AMS', 'ANG', 'BHG', 'GFI', 'GLN', 'IMP', 'NPH', 'SOL', 'SSW']
indi25 = ['ANH', 'APN', 'AVI', 'BAW', 'BID', 'BTI', 'BVT', 'CFR', 'CLS', 'LHC', 'MCG', 'MEI', 'MNP', 'MRP', 'MTN', 'NPN', 'PPH', 'PRX', 'SHP', 'SPP', 'TBS', 'TFG', 'TRU', 'VOD', 'WHL']
fini15 = ['ABG', 'CPI', 'DSY', 'FSR', 'GRT', 'INL', 'INP', 'NED', 'NRP', 'OMU', 'QLT', 'REM', 'RMI', 'RNI', 'SBK', 'SLM']
tech = ['AEL', 'BYI', 'DTC', 'KRO', 'NPN', 'PRX']
telco = ['BLU', 'MCG', 'MTN', 'TKG', 'VOD']
health = ['AIP', 'APN', 'LHC', 'MEI', 'NTC']
fin = ['ABG', 'AFH', 'AIL', 'BAT', 'CML', 'CPI', 'DSY', 'FSR', 'HCI', 'INL', 'INP', 'JSE', 'KST', 'MTM', 'N91', 'NED', 'NY1', 'OMU', 'PSG', 'QLT', 'REM', 'RMI', 'RNI', 'SBK', 'SLM', 'SNT', 'TCP', 'ZED']
real_estate = ['ATT', 'CCO', 'EMI', 'EQU', 'FFA', 'FFB', 'FTB', 'GRT', 'HMN', 'HYP', 'IPF', 'L2D', 'LTE', 'MLI', 'MSP', 'NRP', 'RDF', 'RES', 'SAC', 'SRE', 'SSS', 'VKE']
cons_disc = ['ADH', 'CFR', 'CLH', 'COH', 'CSB', 'FBR', 'ITE', 'MRP', 'MSM', 'MTA', 'MTH', 'PPH', 'SNH', 'SSU', 'SUI', 'TFG', 'TRU', 'TSG', 'WHL']
cons_stap = ['ANH', 'ARL', 'AVI', 'BID', 'BTI', 'CLS', 'DCP', 'DGH', 'LBR', 'OCE', 'PIK', 'RFG', 'SHP', 'SPP', 'TBS']
ind = ['AFT', 'BAW', 'BVT', 'GND', 'HDC', 'KAP', 'MNP', 'MUR', 'PPC', 'RBX', 'RLO', 'SPG', 'TXT', 'WBO']
basic_mat = ['ACL', 'AFE', 'AGL', 'AMS', 'ANG', 'ARI', 'BHG', 'DRD', 'GFI', 'GLN', 'HAR', 'IMP', 'KIO', 'NPH', 'OMN', 'PAN', 'RBP', 'SAP', 'SOL', 'SSW', 'THA']
energy = ['EXX', 'MKR', 'TGA']

#JSE Codes
code_dict = {'Top 40' : 'J200T', 'Mid Cap' : 'J201T', 'Small Cap' : 'J202T', 'Resource 10' : 'J210T', 'Industrial 25' : 'J211T', 'Financial 15' : 'J212T', 'Technology' : 'JI0010T', 'Telecommunications' : 'JI0015T', 'Health Care' : 'JI0020T', 'Financials' : 'JI0030T', 'Real Estate' : 'JI0035T', 'Consumer Discretionary' : 'JI0040T', 'Consumer Staples' : 'JI0045T', 'Industrials' : 'JI0050T', 'Basic Materials' : 'JI0055T', 'Energy' : 'JI0060T'}
index_dict = {'Top 40' : top40, 'Mid Cap' : midcap, 'Small Cap' : smallcap, 'Resource 10' : resi10, 'Industrial 25' : indi25, 'Financial 15' : fini15, 'Technology' : tech, 'Telecommunications' : telco, 'Health Care' : health, 'Financials' : fin, 'Real Estate' : real_estate, 'Consumer Discretionary' : cons_disc, 'Consumer Staples' : cons_stap, 'Industrials' : ind, 'Basic Materials' : basic_mat, 'Energy' : energy}

#TAB 1
with tab1:
    st.subheader("SA Equity Market Returns")
    
    #Return Table
    
    #Select End Date For Calculation
    end_date = st.sidebar.date_input("End Date", value=max(df.index), min_value=min(df.index), max_value=max(df.index))
    end_date = datetime.datetime(end_date.year, end_date.month, end_date.day)

    #Calculate last quarter based on end date
    last_quarter = math.ceil(end_date.month / 3) - 1

    #Calculation of various start dates for return calculations
    w1_start_date = end_date - relativedelta(days=7)
    mtd_start_date = end_date.replace(day=1) - relativedelta(days=1)
    qtd_start_date = end_date.replace(day=1, month=(last_quarter*3)+1) - relativedelta(days=1)
    ytd_start_date = end_date.replace(day=1, month=1) - relativedelta(days=1)
    y1_start_date = end_date - relativedelta(years=1)
    y2_start_date = end_date - relativedelta(years=2)
    y3_start_date = end_date - relativedelta(years=3)
    y5_start_date = end_date - relativedelta(years=5)
    y7_start_date = end_date - relativedelta(years=7)
    y10_start_date = end_date - relativedelta(years=10)

    #List of all start dates for returns table
    date_list = [w1_start_date, mtd_start_date, qtd_start_date, ytd_start_date, y1_start_date, y2_start_date, y3_start_date, y5_start_date, y7_start_date, y10_start_date]

    #Calculate cumulative return table
    returns = []

    for date in date_list:
        x = (df.loc[end_date] / df.loc[date]) - 1
        returns.append(x)

    returns_df = pd.DataFrame()

    for i in range(len(returns)):
        returns_df[i] = returns[i]

    returns_df.columns = ['Week', 'MTD', 'QTD', 'YTD', '1yr', '2yr', '3yr', '5yr', '7yr', '10yr']
    returns_df.index.name = 'Code'
    
    #Annualise returns longer than 1yr
    returns_df['2yr'] = ((1 + returns_df['2yr']) ** (1/2)) - 1
    returns_df['3yr'] = ((1 + returns_df['3yr']) ** (1/3)) - 1
    returns_df['5yr'] = ((1 + returns_df['5yr']) ** (1/5)) - 1
    returns_df['7yr'] = ((1 + returns_df['7yr']) ** (1/7)) - 1
    returns_df['10yr'] = ((1 + returns_df['10yr']) ** (1/10)) - 1

    #Combine return tables and index codes
    ret_df = pd.merge(codes, returns_df, on='Code')
    ret_df.reset_index(drop=True)

    #Formats for styled tables
    formatdict = {'Week': '{:.2%}', 'MTD': '{:.2%}', 'QTD': '{:.2%}', 'YTD': '{:.2%}', '1yr': '{:.2%}', '2yr': '{:.2%}', '3yr': '{:.2%}', '5yr': '{:.2%}', '7yr': '{:.2%}', '10yr': '{:.2%}'}
    cm = 'RdYlGn'
    norm = matplotlib.colors.TwoSlopeNorm(vcenter=0)

    def background_with_norm(s):
        cmap = matplotlib.cm.get_cmap('RdYlGn')
        norm = matplotlib.colors.TwoSlopeNorm(vcenter=0)
        return ['background-color: {:s}'.format(matplotlib.colors.to_hex(c.flatten())) for c in cmap(norm(s.values))]

    
    #Function For Styled Return Table
    def return_table(index_list):
        table_df = ret_df[ret_df['Code'].isin(index_list)]
        table_df = table_df.iloc[pd.Index(table_df['Code']).get_indexer(index_list)]
        table_df = table_df.drop(['Code'], axis=1)
        table_df = table_df.set_index(['Index'])
        return table_df

    table_df = return_table(equity_indices)

    st.table(table_df.style.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])\
                                    .set_properties(**{'text-align' :'center'})\
                                    .hide_columns([])\
                                    .background_gradient(cmap=cm)\
                                    .format(formatdict)
            )
    
    st.markdown("""---""")
    
    #Draw the Line Graph
    time_period = st.radio("Select time period", options=['QTD', 'YTD', '1yr', '3yr'], index=2, horizontal=True, key='tp')
    
    if time_period == 'QTD':
        start_date = qtd_start_date
    if time_period == 'YTD':
        start_date = ytd_start_date
    if time_period == '1yr':
        start_date = y1_start_date
    if time_period == '3yr':
        start_date = y3_start_date
    
    def draw_altair_chart(graph_codes):
        labels = codes.loc[codes['Code'].isin(graph_codes)].set_index('Code').loc[graph_codes]['Index'].to_list()
        chart = index_returns[start_date:end_date][graph_codes]
        chart.iloc[0] = 0   
        chart.set_axis(labels, axis=1, inplace=True)
        chart = ((1 + chart).cumprod() * 100)
        source = chart.reset_index().melt(id_vars=['index'], value_vars=labels, var_name='sector')
        selection = alt.selection_multi(fields=['sector'], bind='legend')
        fig_alt = alt.Chart(source).mark_line().encode(
                    x=alt.X('index', type='temporal', axis=alt.Axis(title="", format = ("%b %Y"))), 
                    y=alt.Y('value', type='quantitative', axis=alt.Axis(title=""), scale=alt.Scale(zero=False)),
                    color=alt.Color('sector:N', legend=alt.Legend(title="Industry Sector")),
                    opacity=alt.condition(selection, alt.value(1), alt.value(0.1)),
                    tooltip=[alt.Tooltip('index', title='Date'), alt.Tooltip('sector', title='Sector'), alt.Tooltip('value', title='Value', format=",.2f")]
                    ).properties(title="Sector Returns", width=800, height=600
                    ).add_selection(selection)
        return fig_alt

    st.altair_chart(draw_altair_chart(index_sectors), use_container_width=True)

#TAB 2
with tab2:
    st.subheader("SA Industry Sector Returns")
    table_df1 = return_table(industry_sectors)

    st.table(table_df1.style.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])\
                                    .set_properties(**{'text-align' :'center'})\
                                    .hide_columns([])\
                                    .background_gradient(cmap=cm)\
                                    .format(formatdict)
            )
    
    st.markdown("""---""")
    
    #Draw the Line Graph
    time_period1 = st.radio("Select time period", options=['QTD', 'YTD', '1yr', '3yr'], index=2, horizontal=True, key='tp1')
    
    if time_period1 == 'QTD':
        start_date = qtd_start_date
    if time_period1 == 'YTD':
        start_date = ytd_start_date
    if time_period1 == '1yr':
        start_date = y1_start_date
    if time_period1 == '3yr':
        start_date = y3_start_date

    st.altair_chart(draw_altair_chart(industry_sectors), use_container_width=True)
    
    st.markdown("""---""")
    st.subheader("Subsector Returns")
    table_df2 = return_table(subsectors)

    st.table(table_df2.style.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])\
                                    .set_properties(**{'text-align' :'center'})\
                                    .hide_columns([])\
                                    .background_gradient(cmap=cm)\
                                    .format(formatdict)
            )

#TAB 3
with tab3:
    #Insert beginning and end dates for calculation
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=max(prices_df.index) - relativedelta(days=7), min_value=min(prices_df.index), max_value=max(prices_df.index), key='sd1')
    with col2:
        end_date = st.date_input("End Date", value=max(prices_df.index), min_value=min(prices_df.index), max_value=max(prices_df.index), key='sd2')
    
    if start_date < end_date:
        pass
    else:
        st.error('Error: End Date must come after Start Date')

    #Convert datetime.date to datetime.datetime format
    end_date = datetime.datetime(end_date.year, end_date.month, end_date.day)
    start_date = datetime.datetime(start_date.year, start_date.month, start_date.day)

    #Calculate share price returns for period
    stock_price_returns = (prices_df.loc[end_date] / prices_df.loc[start_date]) - 1
    stock_returns_df = pd.DataFrame(stock_price_returns, columns=['Return'])
    stock_returns_df.index.name = 'Code'

    top_20 = stock_returns_df.sort_values('Return', ascending=False)[:20]
    top_20.index.name = 'Code'
    top_20 = pd.merge(top_20, stocks_df[['Code', 'Name']], on='Code').set_index(['Code'])
    
    bottom_20 = stock_returns_df.sort_values('Return', ascending=True)[:20]
    bottom_20.index.name = 'Code'
    bottom_20 = pd.merge(bottom_20, stocks_df[['Code', 'Name']], on='Code').set_index(['Code'])

    #Top/Bottom 20 Charts plotted together
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13.33,7.5), dpi=96)
    maxval = max(max((top_20.Return)), abs(min(bottom_20.Return)))

    axes[0].barh(top_20.Name, top_20.Return, color='green')
    axes[1].barh(bottom_20.Name, bottom_20.Return, color='red')

    axes[0].set_xlim([0, maxval])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Return")
    axes[0].xaxis.set_major_formatter(PercentFormatter(xmax=1))
    axes[0].set_frame_on(False)

    axes[1].set_xlim([-maxval, 0])
    axes[1].set_xlabel("Return")
    axes[1].xaxis.set_major_formatter(PercentFormatter(xmax=1))
    axes[1].invert_yaxis()
    axes[1].yaxis.tick_right()
    axes[1].set_frame_on(False)

    fig.suptitle("Capped SWIX Top & Bottom 20 Share Price Moves: "+start_date.strftime("%d %b %Y")+" to "+end_date.strftime("%d %b %Y"), fontsize=16)

    label1 = top_20.Return.round(3)

    for i in range(len(top_20)):
        axes[0].text(x = label1[i], y = i, s="{:.1%}".format(label1[i]), color = 'black', va='center')

    label2 = bottom_20.Return.round(3)

    for i in range(len(bottom_20)):
        axes[1].text(x = label2[i], y = i, s="{:.1%}".format(label2[i]), color = 'black', va='center', horizontalalignment='right')

    st.pyplot(fig, use_container_width=True)
    
    #Treemap 
    st.markdown("""---""")

    #Find Latest Date and Build Capped SWIX
    latest = benchmark.columns.max()
    cswix = pd.DataFrame(benchmark[latest])
    cswix.rename(columns={cswix.columns[0]: "Weight"}, inplace=True)

    benchmark_df = pd.merge(stocks_df[['Code', 'Name', 'ICB_Industry_Name', 'New_ICB_Industry_Name', 'New_ICB_Super_Sector_Name', 'New_ICB_Sector_Name', 'New_ICB_Sub_Sector_Name']], cswix, on='Code')
    benchmark_df = pd.merge(benchmark_df, stock_returns_df, on='Code')
    benchmark_df = benchmark_df.sort_values('Code').reset_index(drop=True)

    #Page Setup
    st.subheader("Capped SWIX Treemap: "+start_date.strftime("%d %b %Y")+" to "+end_date.strftime("%d %b %Y"))

    fig_tree = px.treemap(benchmark_df, path=['New_ICB_Industry_Name', 'Code'], values='Weight', color='Return', 
                          color_continuous_midpoint=0, color_continuous_scale='RdYlGn',  
                          hover_data=['Name', 'Return', 'Weight'],)
    fig_tree.update_traces(hovertemplate='Name: %{customdata[0]} <br>Return: %{color:.2%} <br>Weight: %{value:.2%}')

    st.plotly_chart(fig_tree, use_container_width=True)

#TAB 4
with tab4:
    #Insert beginning and end dates for calculation
    col1, col2, col3 = st.columns(3)
    with col1:
        option = st.selectbox('Choose FTSE/JSE Index:', ('Top 40', 'Mid Cap', 'Small Cap', 'Resource 10', 'Industrial 25', 'Financial 15', 'Technology', 'Telecommunications', 'Health Care', 'Financials', 'Real Estate', 'Consumer Discretionary', 'Consumer Staples', 'Industrials', 'Basic Materials', 'Energy'), index=0)
    with col2:
        start_date = st.date_input("Start Date", value=max(prices_df.index) - relativedelta(days=7), min_value=min(prices_df.index), max_value=max(prices_df.index), key='sd2')
    with col3:
        end_date = st.date_input("End Date", value=max(prices_df.index), min_value=min(prices_df.index), max_value=max(prices_df.index), key='ed2')
    
    if start_date < end_date:
        pass
    else:
        st.error('Error: End Date must come after Start Date')

    #Convert datetime.date to datetime.datetime format
    end_date = datetime.datetime(end_date.year, end_date.month, end_date.day)
    start_date = datetime.datetime(start_date.year, start_date.month, start_date.day)
    
    def draw_stock_returns_chart(jse_index):
        new_df = stocks_df[stocks_df['Code'].isin(index_dict.get(jse_index))].set_index(['Code'])
    
        new_returns = (prices_df.loc[end_date] / prices_df.loc[start_date]) - 1
        new_returns_df = pd.DataFrame(new_returns, columns=['Return'])
        new_returns_df['Code'] = new_returns_df.index
        new_returns_df = new_returns_df.reset_index(drop=True)
    
        jse_index_returns = new_returns_df[new_returns_df['Code'].isin(index_dict.get(jse_index))]
        jse_index_returns = jse_index_returns.reset_index(drop=True)
        jse_index_returns = pd.merge(jse_index_returns, new_df[['Name']], on='Code')
        jse_index_returns = jse_index_returns.sort_values('Return', ascending=False).reset_index(drop=True)
    
        index_val = df[code_dict.get(jse_index)]
        jse_index_return = (index_val.loc[end_date] / index_val.loc[start_date]) - 1
    
        new_fig, ax = plt.subplots(figsize=(13.33,7.5), dpi=96)

        ax.barh(jse_index_returns.Name, jse_index_returns.Return, color=(jse_index_returns.Return >= 0).map({True:'green', False: 'red'}))
        ax.set_title("FTSE/JSE "+str(jse_index)+" Share Price Moves: "+start_date.strftime("%d %b %Y")+" to "+end_date.strftime("%d %b %Y"))
        ax.set_xlabel("Return")
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.invert_yaxis()
        ax.set_frame_on(False)

        label = jse_index_returns.Return.round(3)
    
        for i in range(len(jse_index_returns)):
            if label[i] >= 0:
                ax.text(x = label[i], y = i, s="{:.1%}".format(label[i]), color = 'black', va='center')
            else: 
                ax.text(x = label[i], y = i, s="{:.1%}".format(label[i]), color = 'black', va='center', horizontalalignment='right')

        plt.axvline(jse_index_return, color='black', linestyle='--', label=str(jse_index)+' Return')
        plt.legend()
        return new_fig
    
    st.pyplot(draw_stock_returns_chart(option))
    
#TAB 5
with tab5:
    st.write("Coming Soon")