# Stock App
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import requests, json, re
from parsel import Selector
from itertools import zip_longest

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
    initial_sidebar_state="collapsed",  #expanded
)

##############################################################################
#############################  FUNCTIONS #####################################
##############################################################################
# def get_tickers(ftp_url):
#     r = requests.get(ftp_url)
#     return [entry.partition('|')[0] for entry in r.text.splitlines()]

@st.cache(allow_output_mutation=True)
def get_tickers(date_str):
    date_int = int(date_str) - 5
    tickers = []
    while ("AMZN" not in tickers):
        url = f'https://ftp.nyse.com/NYSESymbolMapping/NYSESymbolMapping_{date_int}.txt'
        r = requests.get(url)
        tickers = [entry.partition('|')[0] for entry in r.text.splitlines()]
        date_int += 1

    return tickers

# @st.cache(allow_output_mutation=True)
def get_ticker_info(stock):
    return yf.Ticker(stock)

@st.cache(allow_output_mutation=True)
def get_names_dict(url):
    r = requests.get(url)
    return {name[1]+" - "+name[5]:name[1] for name in [entry.split('|') for entry in r.text.splitlines()]}


@st.cache(allow_output_mutation=True)
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


def scrape_google_finance(ticker: str):
    params = {
        "hl": "en"  # language
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36",
    }

    html = requests.get(f"https://www.google.com/finance/quote/{ticker}", params=params, headers=headers, timeout=30)
    selector = Selector(text=html.text)

    # where all extracted data will be temporary located
    ticker_data = {
        "ticker_data": {},
        "about_panel": {},
        "news": {"items": []},
        "finance_perfomance": {"table": []},
        "people_also_search_for": {"items": []},
        "interested_in": {"items": []}
    }

    # current price, quote, title extraction
    ticker_data["ticker_data"]["current_price"] = selector.css(".AHmHk .fxKbKc::text").get()
    ticker_data["ticker_data"]["quote"] = selector.css(".PdOqHc::text").get().replace(" • ", ":")
    ticker_data["ticker_data"]["title"] = selector.css(".zzDege::text").get()

    # about panel extraction
    about_panel_keys = selector.css(".gyFHrc .mfs7Fc::text").getall()
    about_panel_values = selector.css(".gyFHrc .P6K39c").xpath("normalize-space()").getall()

    for key, value in zip_longest(about_panel_keys, about_panel_values):
        key_value = key.lower().replace(" ", "_")
        ticker_data["about_panel"][key_value] = value

    # description "about" extraction
    ticker_data["about_panel"]["description"] = selector.css(".bLLb2d::text").get()
    ticker_data["about_panel"]["extensions"] = selector.css(".w2tnNd::text").getall()

    # news extarction
    if selector.css(".yY3Lee").get():
        for index, news in enumerate(selector.css(".yY3Lee"), start=1):
            ticker_data["news"]["items"].append({
                "position": index,
                "title": news.css(".Yfwt5::text").get(),
                "link": news.css(".z4rs2b a::attr(href)").get(),
                "source": news.css(".sfyJob::text").get(),
                "published": news.css(".Adak::text").get(),
                "thumbnail": news.css("img.Z4idke::attr(src)").get()
            })
    else:
        ticker_data["news"]["error"] = f"No news result from a {ticker}."

    # finance perfomance table
    if selector.css(".slpEwd .roXhBd").get():
        fin_perf_col_2 = selector.css(".PFjsMe+ .yNnsfe::text").get()  # e.g. Dec 2021
        fin_perf_col_3 = selector.css(".PFjsMe~ .yNnsfe+ .yNnsfe::text").get()  # e.g. Year/year change

        for fin_perf in selector.css(".slpEwd .roXhBd"):
            if fin_perf.css(".J9Jhg::text , .jU4VAc::text").get():
                perf_key = fin_perf.css(
                    ".J9Jhg::text , .jU4VAc::text").get()  # e.g. Revenue, Net Income, Operating Income..
                perf_value_col_1 = fin_perf.css(".QXDnM::text").get()  # 60.3B, 26.40%..
                perf_value_col_2 = fin_perf.css(".gEUVJe .JwB6zf::text").get()  # 2.39%, -21.22%..

                ticker_data["finance_perfomance"]["table"].append({
                    perf_key: {
                        fin_perf_col_2: perf_value_col_1,
                        fin_perf_col_3: perf_value_col_2
                    }
                })
    else:
        ticker_data["finance_perfomance"]["error"] = f"No 'finence perfomance table' for {ticker}."

    # "you may be interested in" results
    if selector.css(".HDXgAf .tOzDHb").get():
        for index, other_interests in enumerate(selector.css(".HDXgAf .tOzDHb"), start=1):
            ticker_data["interested_in"]["items"].append(discover_more_tickers(index, other_interests))
    else:
        ticker_data["interested_in"]["error"] = f"No 'you may be interested in` results for {ticker}"

    # "people also search for" results
    if selector.css(".HDXgAf+ div .tOzDHb").get():
        for index, other_tickers in enumerate(selector.css(".HDXgAf+ div .tOzDHb"), start=1):
            ticker_data["people_also_search_for"]["items"].append(discover_more_tickers(index, other_tickers))
    else:
        ticker_data["people_also_search_for"]["error"] = f"No 'people_also_search_for` in results for {ticker}"

    return ticker_data


def discover_more_tickers(index: int, other_data: str):
    """
    if price_change_formatted will start complaining,
    check beforehand for None values with try/except and set it to 0, in this function.

    however, re.search(r"\d{1}%|\d{1,10}\.\d{1,2}%" should make the job done.
    """
    return {
        "position": index,
        "ticker": other_data.css(".COaKTb::text").get(),
        "ticker_link": f'https://www.google.com/finance{other_data.attrib["href"].replace("./", "/")}',
        "title": other_data.css(".RwFyvf::text").get(),
        "price": other_data.css(".YMlKec::text").get(),
        "price_change": other_data.css("[jsname=Fe7oBc]::attr(aria-label)").get(),
        # https://regex101.com/r/BOFBlt/1
        # Up by 100.99% -> 100.99%
        "price_change_formatted": re.search(r"\d{1}%|\d{1,10}\.\d{1,2}%",
                                            other_data.css("[jsname=Fe7oBc]::attr(aria-label)").get()).group()
    }


# @st.cache(allow_output_mutation=True)
def get_index_info(index):
    data = scrape_google_finance(index)

    current = data['ticker_data']['current_price'].partition(',')
    current = float(current[0] + current[-1])

    prev = data["about_panel"]['previous_close'].partition(',')
    prev = float(prev[0] + prev[-1])

    return data['ticker_data']['title'], current, prev



##############################################################################
############################ PLOTS ###########################################
##############################################################################

# def px_intraday(d):
#     fig = px.line(d, x=d.index, y="Close", color_discrete_sequence=["lime"],
#                   template="plotly_dark")
#     fig.update_traces(mode="lines", hovertemplate='<i>Price</i>: $%{y:.2f}' +
#                                                   '<br><i>Time</i>: %{x|%H:%M}<br>')
#     fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
#                       plot_bgcolor="rgba(0,0,0,0)")
#     return fig


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
        margin=dict(t=10,b=10,l=10,r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        yaxis=dict(showgrid=False, title={"font":dict(size=24),"text": "Price ($USD)", "standoff": 10}),
        yaxis2=dict(showgrid=False, title={"font":dict(size=24),"text": "Volume", "standoff": 1.5}),
        xaxis=dict(showline=False, title={"font":dict(size=24), "standoff": 10})
    )
    return fig


# def plot_pie(df):
#     df.loc[df['Market Cap'] < 5.e11, "Name"] = "Other"
#     fig = px.pie(df, values='Market Cap', names='Name', title='Market Cap of US Companies')
#     return fig


@st.cache(allow_output_mutation=True)
def instit_pie(ticker, floatShares):
    inst_df = ticker.institutional_holders
    other_row = {"Holder": "Other Institutions", "Shares": floatShares - inst_df.Shares.sum(), "Date Reported": None,
                 "% Out": None, "Value": None}
    other_row = pd.DataFrame(other_row, index=[0])

    inst_df = pd.concat([inst_df, other_row], axis=0)

    inst_df['pct'] = inst_df.Shares / floatShares

    fig = px.pie(inst_df, values="pct", names="Holder", title='Institutional Holders')
    fig.update_layout(
        template="plotly_dark",
        margin=dict(t=25,b=0,l=0,r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


@st.cache(allow_output_mutation=True)
def px_income(df):
    fig = px.scatter(df, x=df.index, y="Net Income", trendline='ols')
    fig.update_traces(marker_size=14, marker_color="magenta")

    fig.update_layout(
        hoverlabel=dict(align="left", bgcolor="rgba(0,0,0,0)"),
        template="plotly_dark",
        margin=dict(t=0,b=0,l=0,r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        yaxis=dict(showline=False, showgrid=True, title={"text": "Net Income ($USD)",
                                                          "font":dict(size=24),
                                                          "standoff": 25}),
        xaxis=dict(showline=False,showgrid=False, title={"standoff": 25})
    )

    return fig

@st.cache(allow_output_mutation=True)
def opt_scatter(df, exp_date):
    df_sum = df.openInterest.sum()
    y = 'volume' if df_sum < 100 else "openInterest"
    y_label = 'Volume' if df_sum < 100 else "Open Interest"
    df["In The Money"] = df.inTheMoney.mask(df.inTheMoney, "In").mask(~df.inTheMoney, "Out")

    fig = px.scatter(df.round(2), x="strike", y=y,
                     color="impliedVolatility", color_continuous_scale=["magenta", 'yellow', 'lime'],
                     range_color=(0, df.impliedVolatility.max()),
                     size='lastPrice', size_max=25,
                     symbol="In The Money", symbol_map={'In': "0", "Out": "x"},
                     # marginal_x="rug",
                     marginal_y="histogram")

    fig.update_layout(
        template="plotly_dark",
        title_text="<b>Strike vs. "+y_label+"          Expiration: "+exp_date+"<b>",
        title_font=dict(size=30),
        title_x=.5,
        coloraxis_colorbar=dict(yanchor="bottom", y=0, len=.75,
                                title={"text": "Implied<br>Volatility (%)",}),
        legend=dict(yanchor="bottom", y=.75),
        margin=dict(t=50,b=0, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(showline=False, showgrid=True, title={"text": y_label,
                                                         "font": dict(size=24),
                                                         "standoff": 25}),

        xaxis=dict(showline=False, showgrid=True, title={"text": "Strike ($USD)",
                                                          "font": dict(size=24),
                                                          "standoff": 25}),
        xaxis2 = dict(showline=False, showgrid=False),
        xaxis3 = dict(showline=False, showgrid=False),
        yaxis2 = dict(showline=False, showgrid=False),
        yaxis3 = dict(showline=False, showgrid=False),
        #     yaxis4=dict(showline=False,showgrid=False),
    )

    return fig

@st.cache(allow_output_mutation=True)
def parse_headers(hdrs):
    isupper = [[element.isupper() for element in hdr] for hdr in hdrs]  # True for every letter is upper
    any_upper = [np.any([element.isupper() for element in hdr]) for hdr in hdrs]  # True for any header with upper
    dx = [None if not _ else arr.index(True) for (_, arr) in
          zip(any_upper, isupper)]  # index for upper letter or None if all lower
    hdrs_list = [e.partition(e[:d]) for (e, d) in zip(hdrs, dx)]  # split headers at capital letter
    words = [[word.capitalize() for word in hdr if word] for hdr in hdrs_list]  # remove empty str and capitalize
    hdrs = [' '.join(word) for word in words]  # join headers
    return ["<b>" + hdr + "</b>" for hdr in hdrs]  # bold headers


@st.cache(allow_output_mutation=True)
def opt_table(df, kind='Call', spread=5):
    df = df.drop(columns=['contractSymbol', 'change', 'currency', 'contractSize', 'lastTradeDate']).round(2)
    dx = max(df[df.inTheMoney].index) if "C" in kind else min(df[df.inTheMoney].index)
    df['color'] = df.inTheMoney.mask(df.inTheMoney, other='rgb(10, 255, 30)').mask(~df.inTheMoney,
                                                                                   other='rgb(255, 45, 10)')
    df = df.loc[dx - spread:dx + spread]
    hdrs = parse_headers(list(df.drop(columns=['color', 'inTheMoney']).columns))

    fig = go.Figure(data=[go.Table(
        columnwidth=[150 for _ in range(len(hdrs))],

        header=dict(values=hdrs,
                    fill_color='rgb(0,0,0,0)',
                    font=dict(color='white', size=14),
                    height=50,
                    align='center'),
        cells=dict(values=[df.strike, df.lastPrice, df.bid, df.ask, df.percentChange,
                           df.volume, df.openInterest, df.impliedVolatility],
                   line_color='white',
                   fill_color=[df.color],
                   font=dict(color='black', size=14),
                   height=25,
                   align='center'))
    ])

    fig.update_layout(
        margin=dict(t=50, b=0, l=10, r=10),
        title_text="<b>"+kind+" Options Chain<b>",
        title_x=.5,
        title_font=dict(size=30))

    return fig
########################################################################################
########################################################################################
#################################### MAIN Code #########################################
########################################################################################
########################################################################################
# TITLE & LOGO:
st.title('💎 **U.S. Stocks App** 💎')
# ":diamonds: :gem:  :fire:"
# ":dollar: :moneybag: :money_with_wings: :fire:"
# st.subheader('The Smart App for Analyzing U.S. Stocks by @ObaiShaikh')
'The Smart App for Analyzing U.S. Stocks'
'By Obai Shaikh'
"---"
########################################################################################
#################################### SIDEBAR ###########################################
########################################################################################
# GET SYMBOLS (NYSE ftp site)
# Method 1: symbol + name
url="https://ftp.nyse.com/Reference%20Data%20Samples/NYSE%20GROUP%20SECURITY%20MASTER/" \
    "NYSEGROUP_US_REF_SECURITYMASTERPREMIUM_EQUITY_4.0_20220927.txt"
ticker_dict = get_names_dict(url)

# # Method 2: symbol only
# date_str = date.today().strftime("%Y%m%d")
# stocks = get_tickers(date_str)
# Posting_Date|Ticker_Symbol|Security_Name|Listing_Exchange|Effective_Date|Deleted_Date|
# Tick_Size_Pilot_Program_Group|Old_Tick_Size_Pilot_Program_Group|Old_Ticker_Symbol|Reason_for_Change

# Method 3: from file
# stocks = list(np.genfromtxt('nasdaqlisted.txt', delimiter='|',skip_header=1,dtype=str)[:,0])
# df = pd.read_csv("nasdaq_screener.csv",index_col=0)
# stocks = list(df.index)
# ['MSFT','AAPL','TSLA','AMZN','BA', 'GOOGL','GOOG','NVDA','MVST','MILE']
########################################################################################
# TODO: ADD LOGO HERE: STOX
# Language input
lang = st.sidebar.radio(
    'Langauge:',
    ('English', 'العربية'))
# Ticker input
stock = st.sidebar.selectbox(
    'Ticker:',
    list(ticker_dict), index=list(ticker_dict.values()).index('AMZN'), key="stock")
stock = ticker_dict[stock]
########################################################################################
#################################### MAIN PAGE #########################################
########################################################################################

############################## Major Market Metrics ####################################
# MAJOR INDEXES (GOOGLE FINANCE):
dj_index, sp_index, nas_index=".DJI:INDEXDJX", ".INX:INDEXSP", ".IXIC:INDEXNASDAQ"

# FUTURES (NOT WORKING):
# dj_fut, sp_fut, nas_fut="YMWOO:CBOT", "ESWOO:CME_EMINIS", "NQWOO:CME_EMINIS"

# Get indexes info (Google Finance)
dj_name, dj_current, dj_prev = get_index_info(dj_index)
sp_name, sp_current, sp_prev = get_index_info(sp_index)
nas_name, nas_current, nas_prev = get_index_info(nas_index)

# Get indexe futures info (Google Finance)
# spf_name, spf_current, spf_prev = get_index_info(sp_fut)

# Print Index Metrics (Streamlit):
dj_col, sp_col, nas_col = st.columns(3)
dj_col.metric(dj_name, f"{dj_current:,}", round(dj_current-dj_prev,2))
sp_col.metric(sp_name, f"{sp_current:,}", round(sp_current-sp_prev,2))
nas_col.metric(nas_name, f"{nas_current:,}", round(nas_current-nas_prev,2))

# djf_col, spf_col, nasf_col = st.columns(3)
# spf_col.metric(spf_name, f"{spf_current:,}", round(spf_current-spf_prev,2))

# overwriting elements in-place
################## Major Market Metrics ############################
# with st.empty():  # overwriting elements in-place
#     for sec in range(5):
#         st.write(f"{sec} seconds have passed")
#         time.sleep(1)
#     st.write("Times up!")

########################################################################################
################################ YAHOO FINANCE #########################################
########################################################################################
"---"
# Ticker input
stock = st.selectbox(
    'Search a stock:',
    list(ticker_dict), index=list(ticker_dict.values()).index('AMZN'), key="stock2")
stock = ticker_dict[stock]
########################################################################################
########################################################################################
# STOCK INFO (Yfinance)
ticker = get_ticker_info(stock)

# LANGUAGE DICT:
lang_dict = get_lang_dict(lang)

# TICKER INFORMATION DICT
idict = ticker.info


########################################################################################
################################# PRICE CHART  #########################################
########################################################################################
with st.container():
    plt_col1, plt_col2, plt_col3 = st.columns([5,1,1],gap="small")

    plt_col1.header(idict['shortName'])
    period = plt_col2.selectbox("Duration:",["1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"],
                                index=0, key="period")
    interval = plt_col3.selectbox("Interval:",['1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo'],
                                index=0,help="fetch data by interval (intraday only if period < 60 days)", key="interval")

    d = ticker.history(period=period, interval=interval,
                       rounding=True).drop(columns=['Dividends', 'Stock Splits'], errors="ignore")

    st.plotly_chart(intraday(d), use_container_width=True)

########################################################################################
################################# TICKER METRICS  ######################################
########################################################################################
# Calculations:
div_yld = 0 if idict["dividendYield"] is None else idict["dividendYield"]
pe = 0 if idict["trailingPE"] is None else idict["trailingPE"]
flabels = ["MARKET CAP", "AVG VOLUME", "P/E RATIO", "DIVIDEND YIELD"]
fmetrics = [idict["marketCap"], idict["averageDailyVolume10Day"], pe, div_yld]
fin_labels = ["REVENUE", "NET INCOME", "OPEX", ]

if 'sector' in idict:
    general_labels = ["SECTOR", "HEADQUARTERS", "EMPLOYEES", "WEBSITE"]
    general_metrics = [idict["sector"], idict["city"] + ", " + idict["country"],
                       idict["fullTimeEmployees"], idict["website"]]
else:
    general_labels = ["CATEGORY", "MARKET", "TIME ZONE"]
    general_metrics = [idict["category"], idict["market"], idict["exchangeTimezoneName"]]

'---'
with st.container():
    st.header(stock + ' Summary')
    # info_container = st.container()
    # info_container2 = st.container()
    columns = st.columns(len(general_labels))
    columns2 = st.columns(len(flabels))

    for col2, flabel, metric2 in zip(columns, flabels, fmetrics):
        col2.caption(flabel)
        col2.markdown(str(millify(metric2)))

    for col, label, metric in zip(columns2, general_labels, general_metrics):
        col.caption(label)
        col.markdown(metric)
        col.empty()

################## Target Price Bar ############################
# recommendation = "RECOMMENDATION"
# idict["recommendationKey"]

#################################################################
####################### OPTIONS #################################
#################################################################
"---"
# with st.container():
with st.expander(stock + ' Options'):

    call_tab, put_tab = st.tabs(["Calls","Puts"])

    with call_tab:
        opt_type="Call"
        opt_col1, opt_col2 = st.columns([4, 1], gap="small")
        opt_col1.header(stock+ "  Calls")
        # opt_type = opt_col2.selectbox(
        #     'Call/Put:',
        #     ["Call", "Put"], index=0, key="option_type")

        exp_date = opt_col2.selectbox(
            'Expiration:',
            ticker.options, index=0, key="call_exp_date")

        opt = ticker.option_chain(exp_date)

        df=opt.calls
        st.plotly_chart(opt_table(df, kind=opt_type), use_container_width=True)
        # "---"
        st.plotly_chart(opt_scatter(df, exp_date), use_container_width=True)

    with put_tab:
        opt_type = "Put"
        opt_col1, opt_col2 = st.columns([4, 1], gap="small")
        opt_col1.header(stock + "  Puts")
        exp_date = opt_col2.selectbox(
            'Expiration:',
            ticker.options, index=0, key="put_exp_date")
        opt = ticker.option_chain(exp_date)

        df=opt.puts
        st.plotly_chart(opt_table(df, kind=opt_type), use_container_width=True)
        # "---"
        st.plotly_chart(opt_scatter(df, exp_date), use_container_width=True)

        # if "Call" in opt_type:
        #     df=opt.calls #.round(2)
        #     st.plotly_chart(opt_table(df, exp_date, kind=opt_type), use_container_width=True)
        #     st.plotly_chart(opt_scatter(df, exp_date), use_container_width=True)
        # else:
        #     df=opt.puts #.round(2)
        #     st.plotly_chart(opt_table(df, exp_date, kind=opt_type), use_container_width=True)
        #     st.plotly_chart(opt_scatter(df, exp_date), use_container_width=True)


########################################################################################
################################ EARNINGS Expander #####################################
########################################################################################
with st.expander(stock + ' Earnings'):

    qtab, ytab = st.tabs(["Quarterly","Yearly"])

    with qtab:
        df = ticker.quarterly_financials.T
        st.plotly_chart(px_income(df), use_container_width=True)

    with ytab:
        df = ticker.financials.T
        st.plotly_chart(px_income(df), use_container_width=True)
########################################################################################
########################## HOLDERS - PIE Expander ######################################
########################################################################################
with st.expander(stock + ' Holders'):
    tab1, tab2 = st.tabs(["Institutions","Insiders"])

    with tab1:
        # st.subheader('Price')
        st.plotly_chart(instit_pie(ticker, idict['floatShares']), use_container_width=True)

    with tab2:
        # st.subheader('Financials')
        st.plotly_chart(instit_pie(ticker, idict['floatShares']), use_container_width=True)
########################################################################################
################################# BALANCE SHEET ########################################
########################################################################################
with st.expander(stock + ' Balance Sheet'):
    qbtab, ybtab = st.tabs(["Quarterly","Yearly"])
    with qbtab:
        df = ticker.quarterly_balance_sheet
        df[df.isna()] = 0.0
        df = pd.DataFrame(df, columns=[col.strftime('%Y-%m-%d') for col in df.columns])

        # df = '$' + (df/1000000).round(1).astype(str) + " Million"
        # df = df.apply(lambda x: ["{:,}".format(x) for _ in df],axis=1, result_type='expand') #.ljust(12))
        for col in qb:
            qb[col] = qb[col].apply(lambda x: "${:,} Million".format(x).ljust(12))

        # df = '$' + df + " Million"
        st.dataframe(df, use_container_width=True)

    with ybtab:
        df = ticker.balance_sheet.round(0)
        df[df.isna()] = 0
        df = pd.DataFrame(df, columns=[col.strftime('%Y-%m-%d') for col in df.columns],dtype=int)
        st.dataframe(df, use_container_width=True)


########################################################################################
##################################### TABLES ###########################################
########################################################################################

# Price:
pinfo = np.round([
    idict['currentPrice'], idict['previousClose'],
    idict['fiftyTwoWeekHigh'], idict['fiftyTwoWeekLow'],
    idict['dayHigh'], idict['dayLow'], idict['volume']], 2)
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
         idict['dateShortInterest'], idict['shortRatio']]

pricelist = [floatToString(s) if s < 1e6 else str(millify(s)) for s in pinfo]
einfo = [0 if x is None else x for x in einfo]  # replace None with 0
elist = [millify(n) for n in einfo]


"---"
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
# with st.container():
#     col1, col2, col3 = st.columns(3)

# 	with col1:
# 		tbl1

# 	with col2:
# 		tbl2
#################################################################

if st.checkbox("Information dict:"):
    idict

    
#################################################################

# st.subheader('Jave for loop:')
# code = '''int array[] = {0, 1, 2, 3, 4};
# string text = "Player ";
# for (int value : array)
# cout << text + std::to_string(value) << endl;'''
#
# st.code(code, language='python')

# show hidden text
if st.checkbox("TODO:"):
    st.text("1. append new data every 1m, update intraday plot")
    st.text("2. add useful resources, youtube links, urls")
    st.text("3. add 1M, 6M, 1Y charts")
    st.text("4. add pie chart + convert tables to metrics")
    st.text("5. add recommendation + quarterly financials, balance sheet, FCF, like google")
    st.text("6. Make price chart live! append data")
    st.text("7. fix millify, AMZN mrkt cap=1.2 Trillion, use log10")
    st.text("8. add twtr, linkedin, github contact info")

    'What does it mean when the lines cross?'
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
