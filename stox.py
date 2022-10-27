# Stock App
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import requests, json, re
from parsel import Selector
from itertools import zip_longest

import requests, ftplib
from io import BytesIO
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
def flatten(mylist):
    return [item for sublist in mylist for item in sublist]

def get_tickers(date_str):
    date_int = int(date_str) - 5
    tickers = []
    while ("AMZN" not in tickers):
        url = f'https://ftp.nyse.com/NYSESymbolMapping/NYSESymbolMapping_{date_int}.txt'
        r = requests.get(url)
        tickers = [entry.partition('|')[0] for entry in r.text.splitlines()]
        date_int += 1
    return tickers


def get_names_dict(url):
    r = requests.get(url)
    return {name[1]+" - "+name[5]:name[1] for name in [entry.split('|') for entry in r.text.splitlines()]}


def nasdaq_df(host='ftp.nasdaqtrader.com', sub_dir="symboldirectory", fname='nasdaqlisted.txt'):
    with ftplib.FTP(host, 'anonymous') as ftp:
        ftp.cwd(sub_dir)
        with BytesIO() as f:
            ftp.retrbinary('RETR ' + fname, f.write)
            f.seek(0)
            df = pd.read_csv(f, sep="|")

    return df


@st.cache(allow_output_mutation=True)
def get_symbols_dict(today):
    '''Get Nasdaq Ticker Symbols (Equity & ETFs)

    Returns a dict-like
    Symbol -> {Name -> Symbol}
    ETF    -> {Name -> if_ETF}
    '''
    # print(f"Downloading U.S. Ticker Symbols  -  Last Updated: {today}")
    with st.spinner(f"Downloading U.S. Ticker Symbols  -  Last Updated: {today}"):
        df = nasdaq_df().loc[:, ["Symbol", "Security Name", "ETF"]].iloc[:-1]  # drop last row is file creation time
        other_df = nasdaq_df(fname="otherlisted.txt").rename(columns={"ACT Symbol": "Symbol"}).loc[:,
                   ["Symbol", "Security Name", "ETF", "Exchange"]].iloc[:-1]

        # Nasdaqlisted
        df["Name"] = df.Symbol + " - " + df['Security Name'].apply(lambda x: f"{x}".split('-')[0].strip())
        df = df.set_index('Symbol').drop(columns=['Security Name'], errors='ignore')
        df["ETF"] = df.ETF.mask(df.ETF.isna(), other='N')
        df['Exchange'] = 'NASDAQ'

        # Otherlisted
        other_df["Name"] = other_df.Symbol + " - " + other_df['Security Name']
        other_df = other_df.set_index('Symbol').drop(columns=['Security Name'], errors='ignore')
        other_df["ETF"] = other_df.ETF.mask(other_df.ETF.isna(), other='N')
        mapper = {'A': "NYSEAMERICAN",
                  'N': "NYSE",
                  'P': "NYSEARCA",
                  'Z': "BATS",
                  'V': "IEXG"
                  }
        other_df["Exchange"] = other_df.Exchange.replace(mapper)
        merged = pd.concat([df, other_df])

    return merged.to_dict()


def get_ticker_info(stock):
    return yf.Ticker(stock)


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


def yf_metrics(idict, isetf):
    # Calculations:
    div_yld = 0 if idict["dividendYield"] is None else idict["dividendYield"]
    # fin_labels = ["REVENUE", "NET INCOME", "OPEX", ]
    smrylbl = ["CURRENT PRICE", "PREV. CLOSE", "HIGH", "LOW"]

    if 'N' in isetf:  # for stocks

        flabels = ["MARKET CAP", "AVG VOLUME", "PEG RATIO", "DIVIDEND YIELD"]
        loclbl = ["SECTOR", "HEADQUARTERS", "EMPLOYEES", "WEBSITE"]

        smry_metrics = np.round([idict['currentPrice'], idict['previousClose'],
                                 idict['dayHigh'], idict['dayLow']],
                                2)  # idict['fiftyTwoWeekHigh'], idict['fiftyTwoWeekLow']
        peg = 0 if idict["pegRatio"] is None else idict["pegRatio"]

        fmetrics = [idict["marketCap"], idict['volume'], peg, div_yld]

        lmetrics = [idict["sector"], idict["city"] + ", " + idict["country"],
                    idict["fullTimeEmployees"], f'[{idict["shortName"]}]({idict["website"]})']


    else:  # for ETFs

        flabels = ["TOTAL ASSETS", "AVG VOLUME", "3YR AVG RETURN", "DIVIDEND YIELD"]
        loclbl = ["CATEGORY", 'EXCHANGE', "MARKET", "TIME ZONE"]

        smry_metrics = np.round([idict['regularMarketPrice'], idict['previousClose'],
                                 idict['dayHigh'], idict['dayLow']],
                                2)  # idict['fiftyTwoWeekHigh'], idict['fiftyTwoWeekLow']
        avg_return = 0 if idict["threeYearAverageReturn"] is None else idict["threeYearAverageReturn"]
        fmetrics = [idict["totalAssets"], idict["volume"], avg_return, div_yld]
        lmetrics = [idict["category"], idict["exchange"], idict["market"], idict["exchangeTimezoneName"]]

    with st.container():
        st.header(stock + ' Summary')
        columns = st.columns(len(smrylbl))
        columns2 = st.columns(len(flabels))
        columns3 = st.columns(len(loclbl))

        for col, label, metric in zip(columns, smrylbl, smry_metrics):
            col.caption(label)
            col.markdown(str(metric))

        for col, label, metric in zip(columns2, flabels, fmetrics):
            col.caption(label)
            col.markdown(str(metric))

        for col, label, metric in zip(columns3, loclbl, lmetrics):
            col.caption(label)
            if isinstance(metric, int):
                metric = f"{metric:,}"
            col.markdown(metric)

    with st.expander("Click here for tips:"):
        st.text("* PEG RATIO : Price/Earnings-to-Growth lower than 1.0 is best, "
                "suggesting that a company is relatively undervalued.  -Investopedia")

    return



def gf_metrics(ginfo, idict, isetf):
    # Calculations:
    div_yld = 0 if "-" in ginfo["dividend_yield"] else ginfo["dividend_yield"]
    # fin_labels = ["REVENUE", "NET INCOME", "OPEX", ]
    smrylbl = ["CURRENT PRICE", "PREV. CLOSE", "HIGH", "LOW"]

    if 'N' in isetf:  # for stocks

        flabels = ["MARKET CAP", "AVG VOLUME", "PE RATIO", "DIVIDEND YIELD"]
        loclbl = ["SECTOR", "HEADQUARTERS", "EMPLOYEES", "WEBSITE"]

        smry_metrics = [ginfo['current_price'], ginfo['previous_close'],
                        "$"+str(idict['dayHigh']), "$"+str(idict['dayLow'])]
        # idict['fiftyTwoWeekHigh'], idict['fiftyTwoWeekLow']
        peg = 0 if "-" in ginfo["p/e_ratio"] else ginfo["p/e_ratio"]

        fmetrics = [ginfo["market_cap"], ginfo['avg_volume'], peg, div_yld]

        lmetrics = [idict["sector"], idict["city"] + ", " + idict["country"],
                    ginfo["employees"], f'[{idict["shortName"]}]({idict["website"]})']


    else:  # for ETFs

        flabels = ["TOTAL ASSETS", "AVG VOLUME", "3YR AVG RETURN", "DIVIDEND YIELD"]
        loclbl = ["CATEGORY", 'EXCHANGE', "MARKET", "TIME ZONE"]

        smry_metrics = [idict['regularMarketPrice'], idict['previousClose'],
                                 idict['dayHigh'], idict['dayLow']]
        # idict['fiftyTwoWeekHigh'], idict['fiftyTwoWeekLow']
        avg_return = 0 if idict["threeYearAverageReturn"] is None else idict["threeYearAverageReturn"]
        fmetrics = [idict["totalAssets"], idict["volume"], avg_return, div_yld]
        lmetrics = [idict["category"], idict["exchange"], idict["market"], idict["exchangeTimezoneName"]]

    with st.container():
        st.header(stock + ' Summary')
        columns = st.columns(len(smrylbl))
        columns2 = st.columns(len(flabels))
        columns3 = st.columns(len(loclbl))

        for col, label, metric in zip(columns, smrylbl, smry_metrics):
            col.caption(label)
            col.markdown(str(metric))

        for col, label, metric in zip(columns2, flabels, fmetrics):
            col.caption(label)
            col.markdown(str(metric))

        for col, label, metric in zip(columns3, loclbl, lmetrics):
            col.caption(label)
            if isinstance(metric, int):
                metric = f"{metric:,}"
            col.markdown(metric)

    # with st.expander("Click here for tips:"):
    #     st.text("* PEG RATIO : Price/Earnings-to-Growth lower than 1.0 is best, "
    #             "suggesting that a company is relatively undervalued.  -Investopedia")

    return


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


def get_index_info(index):
    data = scrape_google_finance(index)

    current = data['ticker_data']['current_price'].partition(',')
    current = float(current[0] + current[-1])

    prev = data["about_panel"]['previous_close'].partition(',')
    prev = float(prev[0] + prev[-1])

    return data['ticker_data']['title'], current, prev


def google_stock_info(google_ticker):
    data = scrape_google_finance(google_ticker)

    info = data['ticker_data'] | data["about_panel"]  # dict
    news = data["news"]["items"]  # list

    return info, news


##############################################################################
############################ PLOTS ###########################################
##############################################################################


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
def instit_pie(ticker):
    inst_df = ticker.institutional_holders
    # other_row = {"Holder": "Other", "Shares": floatShares - inst_df.Shares.sum(), "Date Reported": None,
    #              "% Out": None, "Value": None}
    # other_row = pd.DataFrame(other_row, index=[0])

    # inst_df = pd.concat([inst_df, other_row], axis=0)
    inst_df['pct'] = inst_df.Shares #/ floatShares

    fig = px.pie(inst_df, values="pct", names="Holder")
    fig.update_layout(
        template="plotly_dark",
        margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        # title={'text': 'TOP 10 HOLDERS', "font": dict(size=24)},
    )
    return fig


@st.cache(allow_output_mutation=True)
def etf_holdings_pie(df):
    df['holdingPercent'] = df.holdingPercent.round(2)
    fig = px.pie(df, values="holdingPercent", names="holdingName",
)
    fig.update_layout(
        template="plotly_dark",
        margin=dict(t=25,b=0,l=0,r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title={'text': 'TOP 10 HOLDERS', "font": dict(size=24)}
    )
    return fig


@st.cache(allow_output_mutation=True)
def px_income(df):
    fig = px.scatter(df, x=df.index, y="Net Income", trendline='ols')
    fig.update_traces(marker_size=14, marker_color="magenta")

    fig.update_layout(
        hoverlabel=dict(align="left", bgcolor="rgba(0,0,0,0)"),
        template="plotly_dark",
        height=250,
        margin=dict(t=0,b=0,l=0,r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        yaxis=dict(showline=False, showgrid=True, title={"text": "Net Income ($USD)",
                                                          "font":dict(size=18),
                                                          "standoff": 20}),
        xaxis=dict(showline=False,showgrid=False),
        title={'text': 'PROFITS ', "font": dict(size=24)}

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
def opt_table(df, kind, spread=5):
    # df[df.isnull()] = 0
    df = df.drop(columns=['contractSymbol', 'change', 'currency', 'contractSize', 'lastTradeDate']).round(2)
    dx = max(df[df.inTheMoney].index) if "C" in kind else min(df[df.inTheMoney].index)
    df['color'] = df.inTheMoney.mask(df.inTheMoney,
                                     other='rgb(10, 255, 30)').mask(~df.inTheMoney,
                                                                    other='rgb(255, 45, 10)')

    df = df.loc[dx - spread:dx + spread]
    hdrs = parse_headers(list(df.drop(columns=['color', 'inTheMoney']).columns))

    fig = go.Figure(data=[go.Table(
        columnwidth=[150 for _ in range(len(hdrs))],

        header=dict(values=hdrs,
                    fill_color='rgb(0,0,0,0)',
                    font=dict(color='white', size=12),
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
        title_font=dict(size=24))

    return fig


@st.cache(allow_output_mutation=True)
def opt_scatter(df, exp_date):
    # df_sum = df.openInterest.sum()
    y = 'volume' #if df_sum < 100 else "openInterest"
    y_label = 'Volume' #if df_sum < 100 else "Open Interest"
    df["In The Money"] = df.inTheMoney.mask(df.inTheMoney, "In").mask(~df.inTheMoney, "Out")

    fig = px.scatter(df.round(2), x="strike", y=y,
                     color="impliedVolatility", color_continuous_scale=["magenta", 'yellow', 'lime'],
                     range_color=(0, df.impliedVolatility.max()),
                     size='lastPrice', size_max=25,
                     symbol="In The Money", symbol_map={'In': "0", "Out": "x"},
                     # marginal_x="rug",
                     # marginal_y="histogram"
                     )

    fig.update_layout(
        template="plotly_dark",
        title_text="<b>Strike vs. "+y_label+"          Expiration: "+exp_date+"<b>",
        title_font=dict(size=24),
        title_x=0.5,
        coloraxis_colorbar=dict(yanchor="bottom", y=0, len=0.75,
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


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def bs_df(df):
    df[df.isna()] = 0
    df = pd.DataFrame(df, columns=[col.strftime('%Y-%m-%d') for col in df.columns],
                      dtype=int)
    for col in df:
        df[col] = df[col].apply(lambda x: "${:,}".format(x).ljust(20))

    return df


def options(ticker, opt_type):
    opt_col1, opt_col2 = st.columns([2, 1], gap="small")
    opt_col1.header(stock + "  " + opt_type.upper() + "S")

    exp_date = opt_col2.selectbox(
        'Expiration:',
        ticker.options, index=0, key=opt_type+"_exp_date")
    opt = ticker.option_chain(exp_date)

    df = opt.calls if "C" in opt_type else opt.puts

    st.plotly_chart(opt_table(df, kind=opt_type), use_container_width=True)
    # "---"
    st.plotly_chart(opt_scatter(df, exp_date), use_container_width=True)


def plot_news_item(title, link, source, pub_when, thumb):
    fig = go.Figure()

    # Add axes
    fig.add_trace(
        go.Scatter(x=[0, 100], y=[0, 4],marker_opacity=0,mode='markers')
    )
    # Configure axes
    fig.update_xaxes(
        visible=False,
    )
    # Configure axes
    fig.update_yaxes(
        visible=False,
    )

    # Add box
    fig.add_vrect(x0=0, x1=100, line_width=1, line_color="gray")

    # Add image
    fig.add_layout_image(
        dict(
            source=thumb,
            xref="paper",
            yref="paper",
            x=.93,
            y=0.5,
            xanchor="right",
            yanchor="middle",
            sizex=.75,
            sizey=.75,
            sizing='contain',
        )
    )

    words = flatten([line.split() for line in title.splitlines()])
    n_lines = 1 if len(words) < 8 else 2
    text = "<br>".join([" ".join(line) for line in np.array_split(words, n_lines)])
    font = "Droid Sans"  # "Balto" , "Arial"

    # add news headline text
    fig.add_annotation(
        x=1,
        y=3,
        xref="x",
        yref="y",
        xanchor="left",
        yanchor="middle",
        width=750,
        text=text,
        showarrow=False,
        font=dict(
            family=font,
            size=14,
            color="white",
            # color="rgba(0,0,0,1)",
        ),
        align="left",
        bordercolor='rgba(0,0,0,0)',
        bgcolor='rgba(0,0,0,0)',
        opacity=1,
    )

    # Add source and pub_date text
    fig.add_annotation(
        x=1,
        y=.5,
        xref="x",
        yref="y",
        xanchor="left",
        yanchor="middle",
        width=750,
        text="   ".join((source, f'({pub_when})')),
        showarrow=False,
        font=dict(
            family='Arial',
            size=11,
            color="white",
            # color="rgba(0,0,0,1)",
        ),
        align="left",
        bordercolor='rgba(0,0,0,0)',
        bgcolor='rgba(0,0,0,0)',
        opacity=1,
    )

    # # add clickable link (not working)
    # fig.add_annotation(
    #     x=1,
    #     y=2.5,
    #     xref="x",
    #     yref="y",
    #     xanchor="left",
    #     yanchor="middle",
    #     height=150,
    #     width=1000,
    #     text=f"<a href={link}>link link link</a>",
    #     font=dict(
    #         family='Arial',
    #         size=200,
    #         color="rgba(0,0,0,0)",),
    #     showarrow=False,
    #     align="left",
    #     bordercolor='rgba(0,0,0,0)',
    #     bgcolor='rgba(0,0,0,0)',
    #     opacity=0,
    # )

    # update layout properties
    fig.update_layout(
        height=130,
        width=1000,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(r=0, l=0, b=0, t=0),
    )

    return fig


########################################################################################
########################################################################################
#################################### MAIN Code #########################################
########################################################################################
########################################################################################
# TITLE & LOGO:
# title = '💎 **U.S. Stocks App** 💎'
title = '💎 U.S. Stocks App 💎'
welcome = 'The Smart App for Analyzing U.S. Stocks'' by Obai Shaikh'

# st.title(title)
st.markdown(f"<h1 style='text-align: center; color: white;'>{title}</h1>", unsafe_allow_html=True)
titcol1, titcol2, titcol3 = st.columns([2,2,2], gap="small")
titcol2.write(welcome)
# ":diamonds: :gem:  :fire:"
# ":dollar: :moneybag: :money_with_wings: :fire:"
# st.subheader('The Smart App for Analyzing U.S. Stocks by @ObaiShaikh')

# Session State:
if 'rate' not in st.session_state:
    st.session_state.rate = 30       # refresh rate, seconds
########################################################################################
#################################### SIDEBAR ###########################################
########################################################################################

# # Old Method: symbol + name (NYSE) - fixed file
# url="https://ftp.nyse.com/Reference%20Data%20Samples/NYSE%20GROUP%20SECURITY%20MASTER/" \
#     "NYSEGROUP_US_REF_SECURITYMASTERPREMIUM_EQUITY_4.0_20220927.txt"
# ticker_dict = get_names_dict(url)

# # Method 2: symbol only
# date_str = date.today().strftime("%Y%m%d")
# stocks = get_tickers(date_str)
# Posting_Date|Ticker_Symbol|Security_Name|Listing_Exchange|Effective_Date|Deleted_Date|
# Tick_Size_Pilot_Program_Group|Old_Tick_Size_Pilot_Program_Group|Old_Ticker_Symbol|Reason_for_Change
########################################################################################
# TODO: ADD LOGO HERE: STOX
# Language input
lang = st.sidebar.radio(
    'Langauge:',
    ('English', 'العربية'))

########################################################################################
#################################### MAIN PAGE #########################################
########################################################################################
############################## Major Market Metrics ####################################
################################ GOOGLE FINANCE ########################################
"---"
# MAJOR INDEXES (GOOGLE FINANCE):
dj_index, sp_index, nas_index=".DJI:INDEXDJX", ".INX:INDEXSP", ".IXIC:INDEXNASDAQ"

# FUTURES (NOT WORKING):
# dj_fut, sp_fut, nas_fut="YMWOO:CBOT", "ESWOO:CME_EMINIS", "NQWOO:CME_EMINIS"

# Get indexes info (Google Finance)
dj_name, dj_current, dj_prev = get_index_info(dj_index)
sp_name, sp_current, sp_prev = get_index_info(sp_index)
nas_name, nas_current, nas_prev = get_index_info(nas_index)

# Get indexes futures info (Google Finance)
# spf_name, spf_current, spf_prev = get_index_info(sp_fut)

with st.container():
    # st.subheader("Major Markets")
    st.markdown("<h1 style='text-align: center; color: white;'>Major Markets</h1>", unsafe_allow_html=True)

    # Print Index Metrics (Streamlit):
    dj_col, sp_col, nas_col = st.columns(3)
    dj_col.metric(dj_name, f"{dj_current:,}", round(dj_current-dj_prev,2))
    sp_col.metric(sp_name, f"{sp_current:,}", round(sp_current-sp_prev,2))
    nas_col.metric(nas_name, f"{nas_current:,}", round(nas_current-nas_prev,2))

    # djf_col, spf_col, nasf_col = st.columns(3)
    # spf_col.metric(spf_name, f"{spf_current:,}", round(spf_current-spf_prev,2))
########################################################################################
################################ NASDAQ LISTED #########################################
# GET SYMBOLS
today = date.today().strftime("%D")
ticker_etf_dict = get_symbols_dict(today)  # cache-missed once per day
ticker_dict = ticker_etf_dict['Name']
etf_dict = ticker_etf_dict['ETF']
exchange_dict = ticker_etf_dict['Exchange']
########################################################################################
################################# PRICE CHART  #########################################
########################################################################################
"---"
with st.container():
    st.markdown("<h1 style='text-align: center; color: white;'>Stocks Dashboard</h1>", unsafe_allow_html=True)

    plt_col1,_, plt_col2, plt_col3, rfrsh_col = st.columns([3,1,1,1,1],gap="small")
    # Ticker input
    stock = plt_col1.selectbox(
        'Search a stock:',
        list(ticker_dict.values()), index=list(ticker_dict).index('AMZN'), key="stock")
    stock = stock.split('-')[0].strip() # FROM: stock=(short Name)    TO: stock=Symbol (4-letter)
    isetf = etf_dict[stock]
    exchange = exchange_dict[stock]
    ########################################################################################
    # STOCK INFO (Yfinance)
    ticker = get_ticker_info(stock)
    # LANGUAGE DICT:
    lang_dict = get_lang_dict(lang)
    # TICKER INFORMATION DICT
    idict = ticker.info

    st.header(idict['shortName'])
    period = plt_col2.selectbox("Duration:",["1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"],
                                index=0, key="period")
    interval = plt_col3.selectbox("Interval:",['1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo'],
                                index=0,help="fetch data by interval (intraday only if period < 60 days)", key="interval")

    d = ticker.history(period=period, interval=interval,
                       rounding=True).drop(columns=['Dividends', 'Stock Splits'], errors="ignore")

    st.plotly_chart(intraday(d), use_container_width=True)

########################################################################################
################################ GOOGLE FINANCE ########################################
gticker=stock+":"+exchange
ginfo, news = google_stock_info(gticker)
########################################################################################
################################# TICKER METRICS  ######################################
# "---"
# yf_metrics(idict, isetf)

"---"
gf_metrics(ginfo, idict, isetf)
"---"

################## Target Price Bar ############################
# recommendation = "RECOMMENDATION"
# idict["recommendationKey"]



########################################################################################
################################ EARNINGS Expander #####################################
########################################################################################
with st.expander(stock + ' Financial Health'):
    qtab, ytab = st.tabs(["Quarterly", "Yearly"])

    with qtab:
        df = ticker.quarterly_financials.T
        if not df.empty:
            st.plotly_chart(px_income(df), use_container_width=True)

    with ytab:
        df = ticker.financials.T
        if not df.empty:
            st.plotly_chart(px_income(df), use_container_width=True)


#################################################################
####################### OPTIONS #################################
#################################################################
# with st.container():
with st.expander(stock + ' Options'):

    call_tab, put_tab = st.tabs(["Calls", "Puts"])

    with call_tab:
        opt_type="Call"
        options(ticker, opt_type)

    with put_tab:
        opt_type = "Put"
        options(ticker, opt_type)

########################################################################################
########################## HOLDERS - PIE Expander ######################################
########################################################################################
with st.expander(stock + ' Holders'):

    if 'N' in isetf:
        tab1, tab2 = st.tabs(["Institutions", "Insiders"])
        with tab1:
                st.plotly_chart(instit_pie(ticker), use_container_width=True)
        with tab2:
                st.plotly_chart(instit_pie(ticker), use_container_width=True)

    else:
        tab1, tab2 = st.tabs(["Holdings", "Insiders"])
        df = pd.DataFrame(idict['holdings'])
        with tab1:
                st.plotly_chart(etf_holdings_pie(df), use_container_width=True)
        with tab2:
                st.plotly_chart(etf_holdings_pie(df), use_container_width=True)

########################################################################################
################################# BALANCE SHEET ########################################
########################################################################################
with st.expander(stock + ' Balance Sheet'):
    qbtab, ybtab = st.tabs(["Quarterly", "Yearly"])
    with qbtab:
        df = (ticker.quarterly_balance_sheet // 1000000)
        if not df.empty:
            st.dataframe(bs_df(df), use_container_width=True)
            st.text("* in Millions")
    with ybtab:
        df = (ticker.balance_sheet // 1000000)
        if not df.empty:
            st.dataframe(bs_df(df), use_container_width=True)
            st.text("* in Millions")

########################################################################################
###################################### NEWS ############################################
########################################################################################
'---'
with st.container():
    st.subheader("TOP NEWS")
    for item in news:
        st.plotly_chart(plot_news_item(*list(item.values())[1:]), use_container_width=True)

########################################################################################
##################################### TESTING ##########################################
########################################################################################
ginfo
# n_cols = 10
# df1 = pd.DataFrame(
#     np.random.randn(50,n_cols),
#     columns=('col %d' % i for i in range(n_cols))
# )
#
# # my_table = st.table(df1)
#
# df2 = pd.DataFrame(
#     np.random.randn(50,n_cols),
#     columns=('col %d' % i for i in range(n_cols))
# )
#
# # my_table.add_rows(df2)
#
# "^"*50
# st.write("Charts:")
# my_chart = st.line_chart(df1)
# 'sleeping...'
# time.sleep(10)
# my_chart.add_rows(df2)
# 'Done....'

# overwriting elements in-place
################## Major Market Metrics ############################
# with st.empty():  # overwriting elements in-place
#     for sec in range(5):
#         st.write(f"{sec} seconds have passed")
#         time.sleep(1)
#     st.write("Times up!")

########################################################################################
##################################### TABLES ###########################################
########################################################################################
if st.checkbox('Tables (has error):'):
    '''
# Price:
if idict["quoteType"] == "ETF":
    price = idict['regularMarketPrice']
    shares = idict['equityHoldings']['priceToEarnings']
    cash = idict['equityHoldings']['priceToCashflow']
    info4 = idict['equityHoldings']['priceToSales']

else:
    price = idict['currentPrice']
    shares = idict['floatShares']
    cash = idict['freeCashflow']
    info4 = idict['ebitda']

pinfo = np.round([
    price, idict['previousClose'],
    idict['fiftyTwoWeekHigh'], idict['fiftyTwoWeekLow'],
    idict['dayHigh'], idict['dayLow'], idict['volume']], 2)
# Financials
einfo = [
    idict['marketCap'], shares, info4,
    cash, idict['totalDebt'], idict['totalCash'],
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
'''

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
    st.text("9. earnings: estimated vs. actual: surprise")

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

# "---"
# st.info(f"This Page Automatically Reloads Every {st.session_state.rate} Seconds. "
#         f"You Change The Rate Below.")
# c1,c2=st.columns([1,2])
# st.session_state.rate = c1.number_input('Refresh Rate (seconds):', min_value=10, max_value=360, value=30,
#                                        step=10, key='reload_rate')
"---"
with st.container():
    st.subheader("Get in touch:")
    images =['./images/GitHub-Mark-Light.png', './images/LI-In-Bug.png', './images/Twitter-logo.png']
    site_names =['GitHub', 'LinkedIn','Twitter']
    links = ['https://github.com/shaikhon','https://www.linkedin.com/in/obai-shaikh/','https://twitter.com/ObaiShaikh']
    for cc, image, site, link in zip(st.columns(6), images, site_names, links):
        cc.image(image, use_column_width=False, width=70)
        cc.markdown(f"[{site}]({link})")

if rfrsh_col.button('Refresh', help="You can also refresh by pressing 'R'"):
    st.experimental_rerun()
# time.sleep(st.session_state.rate)
# st.experimental_rerun()