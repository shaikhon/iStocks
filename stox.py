# Stock App
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
# import altair as alt
# import plotly.figure_factory as ff
import requests
import math
import time
from datetime import date
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="STOX APP",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)


##################  FUNCTIONS ############################


def get_tickers(ftp_url):
    r = requests.get(ftp_url)
    return [entry.partition('|')[0] for entry in r.text.splitlines()]


@st.cache(allow_output_mutation=True)
def get_ticker_info(stock):
    return yf.Ticker(stock)


def get_lang_dict(lang):
    if lang == 'English':
        lang_dict = {0: 'Current Price',
                     1: 'Prev. Close',
                     2: '52-week High',
                     3: '52-week Low',
                     4: 'Day High',
                     5: 'Day Low',
                     6: 'Volume'}
    else:
        # arabic dict
        lang_dict = {0: 'السعر الآن',
                     1: 'Prev. Close',
                     2: '52-week High',
                     3: '52-week Low',
                     4: 'Day High',
                     5: 'Day Low',
                     6: 'Volume'}
        return lang_dict


def floatToString(inputValue):
    return ('%6.2f' % inputValue).rstrip('0').rstrip('.')


def millify(n):
    millnames = ['', ' Thousand', ' Million', ' Billion', ' Trillion']
    n = float(n)
    millidx = max(0, min(len(millnames) - 1,
                         int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))))

    return '{:.0f}{}'.format(n / 10 ** (3 * millidx), millnames[millidx])


#################### PLOTS #################################################


def px_intraday(d):
    fig = px.line(d, x=d.index, y="Close", color_discrete_sequence=["lime"],
                  template="plotly_dark")
    fig.update_traces(mode="lines", hovertemplate='<i>Price</i>: $%{y:.2f}' +
                                                  '<br><i>Time</i>: %{x|%H:%M}<br>')
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    return fig


def intraday(d):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(x=d.index, y=d["Volume"], opacity=.65,
                         marker={
                             "color": "magenta",  # "#0FCFFF"
                         },
                         hovertemplate='<i>Volume</i>: %{y:,}<extra></extra>'
                         ), secondary_y=True)

    fig.add_trace(go.Scatter(mode="lines", x=d.index, y=d["Close"],
                             line={"color": "lime", "width": 2, },
                             hovertemplate='<i>Price</i>: $%{y:.2f}' + '<br><i>Time</i>: %{x|%H:%M}<br><extra></extra>',
                             ),
                  secondary_y=False)
    # limegreen, lime, #E1FF00, #ccff00

    fig.update_layout(
        hoverlabel=dict(align="left", bgcolor="rgba(0,0,0,0)"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        yaxis=dict(showgrid=False, title={"text": "Price ($)", "standoff": 1}),
        yaxis2=dict(showgrid=False, title={"text": "Volume", "standoff": 1}),
        xaxis=dict(showline=False)
    )
    return fig


def plot_pie(df):
    df.loc[df['Market Cap'] < 5.e11, "Name"] = "Other"
    fig = px.pie(df, values='Market Cap', names='Name', title='Market Cap of US Companies')
    return fig


########################################################################################
#################################### MAIN Code #########################################
########################################################################################
########################################################################################
st.title('**STOX APP**')
st.markdown('Welcome to the intelligent "Due Diligence" app for US companies by @ObaiShaikh')
st.write("TODO:")
st.text("1. add major market metrics, nasdaq, SP500, dow jones on top, ticker metrics bottom")
st.text("2. add useful resources, youtube links, urls")
st.text("3. add 1M, 6M, 1Y charts")
st.text("4. add pie chart + convert tables to metrics")
st.text("5. add recommendation + quarterly financials, balance sheet, FCF, like google")
st.text("6. Make price chart live! append data")
########################################################################################
# Get Ticker symbols from NYSE ftp site
date_str = date.today().strftime("%Y%m%d")
url = f'https://ftp.nyse.com/NYSESymbolMapping/NYSESymbolMapping_{date_str}.txt'
stocks = get_tickers(url)
# stocks = list(np.genfromtxt('nasdaqlisted.txt', delimiter='|',skip_header=1,dtype=str)[:,0])
# df = pd.read_csv("nasdaq_screener.csv",index_col=0)
# stocks = list(df.index)
# ['MSFT','AAPL','TSLA','AMZN','BA', 'GOOGL','GOOG','NVDA','MVST','MILE']
########################################################################################
#################################### SIDEBAR ###########################################
########################################################################################
lang = st.sidebar.radio(
    'Langauge:',
    ('English', 'العربية'))

stock = st.sidebar.selectbox(
    'Ticker:',
    stocks, index=stocks.index('AMZN'))

# STOCK INFO (Yfinance)
ticker = get_ticker_info(stock)


lang_dict = get_lang_dict(lang)

# information dict
idict = ticker.info

# Price:
pinfo = np.round([
    idict['currentPrice'], idict['previousClose'],
    idict['fiftyTwoWeekHigh'], idict['fiftyTwoWeekLow'],
    idict['dayHigh'], idict['dayLow'], idict['volume']
], 2)
# Financials
einfo = [
    idict['marketCap'], idict['floatShares'], idict['ebitda'],
    idict['freeCashflow'], idict['totalDebt'], idict['totalCash'],
    idict['totalRevenue']]  # idict['operatingCashflow'],
# stock info
oinfo = [idict['averageDailyVolume10Day'], idict['fullTimeEmployees']]
# long info
linfo = [
    idict['earningsGrowth'], idict['earningsQuarterlyGrowth'], idict['pegRatio'],
    idict['totalCashPerShare'], idict['revenuePerShare']]
# short interest
sinfo = [idict['debtToEquity'], idict['shortPercentOfFloat'],
         idict['sharesShort'], idict['shortRatio'],
         idict['sharesShortPreviousMonthDate'], idict['sharesShortPriorMonth'],
         idict['dateShortInterest']]

einfo = [0 if x is None else x for x in einfo]  # replace None with 0
pricelist = [floatToString(s) if s < 1e6 else str(millify(s)) for s in pinfo]
elist = [millify(n) for n in einfo]

################## Major Market Metrics ############################
# with st.empty():  # overwriting elements in-place
#     for sec in range(5):
#         st.write(f"{sec} seconds have passed")
#         time.sleep(1)
#     st.write("Times up!")

################## CHART CONTAINER ############################
with st.container():
    st.header(idict['shortName'])
    # intraday - America/New_York
    d = ticker.history(period="1d", interval='1m',
                       rounding=True).drop(columns=['Dividends', 'Stock Splits'], errors="ignore")
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     st.pyplot(plot_intraday(d))
    # st.plotly_chart(px_intraday(d))
    st.plotly_chart(intraday(d), use_container_width=True)

################## Ticker Metrics ############################
general_labels = ["SECTOR", "HEADQUARTERS", "EMPLOYEES", "WEBSITE"]
general_metrics = [idict["sector"], idict["city"] + ", " + idict["country"],
                   idict["fullTimeEmployees"], idict["website"]]

div_yld = 0 if idict["dividendYield"] is None else idict["dividendYield"]
pe = 0 if idict["trailingPE"] is None else idict["trailingPE"]
flabels = ["MARKET CAP", "AVG VOLUME", "P/E RATIO", "DIVIDEND YIELD"]
fmetrics = [idict["marketCap"], idict["averageDailyVolume10Day"], pe, div_yld]

fin_labels = ["REVENUE", "NET INCOME", "OPEX", ]

info_container = st.container()
columns = info_container.columns(len(general_labels))
for col, label, metric in zip(columns, general_labels, general_metrics):
    col.caption(label)
    col.markdown(metric)
    col.empty()
# col.metric(label, metric)

info_container2 = st.container()
columns2 = info_container2.columns(len(flabels))
for col2, flabel, metric2 in zip(columns2, flabels, fmetrics):
    col2.caption(flabel)
    col2.markdown(metric2)
# col2.metric(label, metric)

################## Target Price Bar ############################
recommendation = "RECOMMENDATION"
idict["recommendationKey"]

################## PIE CONTAINER ############################
# with st.container():
#     st.header("Nasdaq Pie")
#     st.plotly_chart(plot_pie(df))

################## TABLE CONTAINER ############################
st.header(stock + ' Summary')
with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader('Price')
        tbl1 = pd.Series(pricelist,
                         index=list(lang_dict.values()), name='$ USD')
        tbl1

    with col2:
        st.subheader('Financials')
        tbl2 = pd.Series(elist,
                         index=['Market Cap', 'Floating Shares', 'EBITDA', 'Free Cash Flow', 'Total Debt', 'Total Cash',
                                'Total Revenue'], name='$ USD')
        tbl2

################## TABLE CONTAINER ############################
with st.container():
    col1, col2, col3 = st.columns(3)

# 	with col1:
# 		tbl1

# 	with col2:
# 		tbl2

#################################################################

st.subheader('Jave for loop:')
code = '''int array[] = {0, 1, 2, 3, 4};
string text = "Player ";
for (int value : array)
cout << text + std::to_string(value) << endl;'''

st.code(code, language='python')

# show hidden text
if st.checkbox('What does it mean when the lines cross?'):
    '# MAIN FEATURES:'
    '1. discounted FCF analysis: new money vid'
    '2. same-sector correlation analysis'
    '3. major metrics + explanations: ratios, boolinger MA, etc.'
    '4. buy/sell signal'
    '5. historical scenarios: assess tool performance'
    '6. sentiment analysis: tweets, news'
    '7. Volume weighted avg. price (VWAP)'

    '## METRICS:'
    '> Graph KPIs & Green/red coded'
    '> KPIs: PE ratio, debttoequity, shortinterest '
    '> CPI (consumer price index): source: new money https://www.youtube.com/watch?v=iOLA9vCFLe0'
    '> CPI minus food and energy: less volatile, removes short-term factors that wont matter few years'
    '  from now'

    '## YF ANALYSIS:'
    'ticker.info = numberOfAnalystOpinions, targetMeanPrice'

    'The 5day moving average lines cross it indicates so and so effect'
    'Also, add trading view TA'
    'Growth Stocks are expected to outperform the market over time'
    'Value Stocks are companies trading below their actual "worth" value'

    '## Other Features: '
    '* chart default: ticker at 1m interval, with 5 ma'
    '* user select: different intervals, different MAs'
    '* highlighting/coloring certain values'
    '* options: language, show hints, '
    '* sidebar as form'
    '* latest stock news'
    '* latest related tweets'
    '* discounted free cash flow analysis: new money vid'
    '* metric performance vs. time: PE ratio graph'

    'check out Trading Signals: https://www.investopedia.com/terms/t/trade-signal.asp'
    'bollinger Moving Avg.: https://www.investopedia.com/terms/b/bollingerbands.asp'

    ########################################

    'DCF:  The purpose of DCF analysis is to estimate the money an investor would receive'
    'from an investment, adjusted for the time value of money. The time value of money'
    'assumes that a dollar today is worth more than a dollar tomorrow because it can be'
    'invested. As such, a DCF analysis is appropriate in any situation wherein a person'
    'is paying money in the present with expectations of receiving more money in the future. -investopedia'
