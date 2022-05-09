import bitfinex
import datetime
import time
import pandas as pd

# function: mine data from bitfinex regarding crypto prices

# launch bitfinex api
api_v2 = bitfinex.bitfinex_v2.api_v2()

# list of all crypto prices of interest
pairs = ['DOTUSD',
        'BTCUSD',
        'ETHUSD',
        'SOLUSD',
        'XRPUSD',
        'LTCUSD',
        'AXSUSD',
        'ZRXUSD']

# preset time intervals of price data - bitfinex only offers 1h time intervals as a minimum
TIMEFRAME = '1h'

# use datatime to assign a start date and end date for data aquisition
t_start = datetime.datetime(2021, 9, 1, 0, 0, 0)
t_start = time.mktime(t_start.timetuple()) * 1000

t_stop = datetime.datetime(2021, 10, 31, 0, 0, 0)
t_stop = time.mktime(t_stop.timetuple()) * 1000

# iterate over each coin of interest and aquire information regarding the transaction
for pair in pairs:
    coinname = pair.split('U')[0]
    result = api_v2.candles(symbol=pair, interval=TIMEFRAME, limit=1000, start=t_start, end=t_stop)

    names = ['Date', 'Open', 'Close', 'High', 'Low', 'Volume']
    df = pd.DataFrame(result, columns=names)
    # save the coin information to a csv file for RNN network
    df.to_csv(f'{coinname}-price-SEPT1-OCT31.csv')