
import numpy as np
import pandas as pd

# sma function
def SMA(data, period=30, column='close'):
    return data[column].rolling(window=period).mean()
    
# create function to signal when to buy and sell an asset
def SMACrossoverSignals(data, SMA1, SMA2, column='close'):
    
    buyList = []
    sellList = []
    flag = -1
    for i in range(0, len(data)):
        if SMA1[i] > SMA2[i]:
            sellList.append(np.nan)
            if flag != 1:
                buyList.append(data[column][i])
                flag = 1
            else:
                buyList.append(np.nan)
        elif SMA1[i] < SMA2[i]:
            buyList.append(np.nan)
            if flag != 0:
                sellList.append(data[column][i])
                flag = 0
            else:
                sellList.append(np.nan)
        else:
            buyList.append(np.nan)
            sellList.append(np.nan)
    return [buyList, sellList]

def get_pips(data, TAKE_PROFIT = 0.2, STOP_LOSS = 0.2): 
    # Pre-allocate memory for the profit_loss array
    profit_loss = np.zeros(data.shape[0])
    
    # Find the indices where the buy signal and take profit conditions are met
    buy_indices = np.where(data['Buy_Signal_Price'] > 0)[0]
    tp_indices = np.where(data['Buy_Signal_Price'] + TAKE_PROFIT <= data['close'])[0]
    # Calculate the profit_loss at the take profit indices
    profit_loss[tp_indices] = TAKE_PROFIT
    
    # Find the indices where the buy signal and stop loss conditions are met
    sl_indices = np.where(data['Buy_Signal_Price'] - STOP_LOSS >= data['close'])[0]
    
    # Calculate the profit_loss at the stop loss indices
    profit_loss[sl_indices] = -STOP_LOSS
    
    # Calculate the total profit_loss
    total_profit_loss = profit_loss.sum()
    
    # get an index of the winning trades
    win_trades_index = np.where(profit_loss > 0)[0]
    # get an index of the losing trades
    loss_trades_index = np.where(profit_loss < 0)[0]    
    # print([total_profit_loss, len(win_trades_index), len(loss_trades_index)])
    # make array of return values with total profit, number of winning trades, and number of trades
    return [round(total_profit_loss,2), len(win_trades_index), len(loss_trades_index)]

def StopLossOptimization(train, take_profit_range, stop_loss_range):
    # Use NumPy's meshgrid function to generate all combinations of take profit and stop loss values
    take_profit_values, stop_loss_values = np.meshgrid(take_profit_range, stop_loss_range)

    # Flatten the arrays to 1D arrays
    take_profit_values = take_profit_values.flatten()
    stop_loss_values = stop_loss_values.flatten()

    # Create a DataFrame with the take profit and stop loss values
    df = pd.DataFrame({'take_profit': take_profit_values.round(2), 'stop_loss': stop_loss_values.round(2)})

    # Use the apply method to calculate the pips and trade win rate for each combination of take profit and stop loss
    vals = df.apply(lambda row: get_pips(train, TAKE_PROFIT=row['take_profit'], STOP_LOSS=row['stop_loss']), axis=1)
    df["pips"] = vals.apply(lambda x: x[0])
    df["wins"] = vals.apply(lambda x: x[1])
    df["losses"] = vals.apply(lambda x: x[2])

    # Pivot the DataFrame to create the final table
    matrix = df.pivot(index='stop_loss', columns='take_profit', values='pips')
    return matrix, df