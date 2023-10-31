#!/usr/bin/env python
# coding: utf-8

# # Import Module

# In[1]:


import numpy as np
import pandas as pd
import warnings; warnings.simplefilter('ignore') 
# import warnings;warnings.filterwarnings('ignore') 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns;sns.set()
# import tushare as ts 
# from WindPy import w;w.start() 
import math
# import talib as ta
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from tqdm import *
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# import xgboost as xgb 
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from statsmodels.tsa.stattools import grangercausalitytests


# In[2]:


import tushare as ts
pro = ts.pro_api('00cd2bab356104dc83c866d0fc39c3c2b2b5b6c016435676304ad62f')


# # Import Data

# ## Tech Factor

# ### Williams

# In[3]:


def WR(data, N1, N2):
    
    
    
    
    data['WR10'] = 100 * (data['high'].rolling(N1).max() - data['close']) / (data['high'].rolling(N1).max() - data['low'].rolling(N1).min())
    data['WR6'] = 100 * (data['high'].rolling(N2).max() - data['close']) / (data['high'].rolling(N2).max() - data['low'].rolling(N2).min())
    
    data['WR10'].fillna(value=100 * (data['high'].expanding().max() - data['close']) / (data['high'].expanding().max() - data['low'].expanding().min()), inplace=True)
    data['WR6'].fillna(value=100 * (data['high'].expanding().max() - data['close']) / (data['high'].expanding().max() - data['low'].expanding().min()), inplace=True)

    return data


# ### MACD

# In[4]:


def MACD(data):

    '''计算MACD指标
        输入参数：symbol <- str      标的代码 （2005年以前上市的不可用）
                start_time <- str  起始时间
                end_time <- str    结束时间
        输出数据：
                macd <- dataframe  macd指标，包括DIFF、DEA、MACD
    '''
    
    # 计算EMA(12)和EMA(16)
    data['EMA12'] = data['close'].ewm(alpha=2 / 13, adjust=False).mean()
    data['EMA26'] = data['close'].ewm(alpha=2 / 27, adjust=False).mean()

    # 计算DIFF、DEA、MACD
    data['DIFF'] = data['EMA12'] - data['EMA26']
    data['DEA'] = data['DIFF'].ewm(alpha=2 / 10, adjust=False).mean()
    data['MACD'] = 2 * (data['DIFF'] - data['DEA'])

    # 上市首日，DIFF、DEA、MACD均为0
    data['DIFF'].iloc[0] = 0
    data['DEA'].iloc[0] = 0
    data['MACD'].iloc[0] = 0

    return data


# ### RSI

# In[5]:


def RSI(data, N1, N2, N3):
    
    
    
    
    data['change'] = data['close'] - data['pre_close']          
    data.loc[(data['pre_close'] == 0), 'change'] = 0            
    data['x'] = data['change'].apply(lambda x: max(x, 0))       
    data['RSI6'] = data['x'].ewm(alpha=1 / N1, adjust=False).mean() / (np.abs(data['change']).ewm(alpha=1/N1, adjust=False).mean()) * 100
    data['RSI12'] = data['x'].ewm(alpha=1 / N2, adjust=False).mean() / (np.abs(data['change']).ewm(alpha=1 / N2, adjust=False).mean()) * 100
    data['RSI24'] = data['x'].ewm(alpha=1 / N3, adjust=False).mean() / (np.abs(data['change']).ewm(alpha=1 / N3, adjust=False).mean()) * 100
    del data['x']
    
    return data


# ### KDJ

# In[6]:


def KDJ(data, N, M1, M2):
    
    
    
    lowList = data['low'].rolling(N).min()
    lowList.fillna(value=data['low'].expanding().min(), inplace=True)
    highList = data['high'].rolling(N).max()
    highList.fillna(value=data['high'].expanding().max(), inplace=True)
    
    rsv = (data['close'] - lowList) / (highList - lowList) * 100
    
    data['kdj_k'] = rsv.ewm(alpha=1/M1, adjust=False).mean()     
    data['kdj_d'] = data['kdj_k'].ewm(alpha=1/M2, adjust=False).mean()
    data['kdj_j'] = 3.0 * data['kdj_k'] - 2.0 * data['kdj_d']
    
    return data


# ### DMA

# In[7]:


def DMA(data, N1, N2, M):
    
    
    
    data['MA10'] = data['close'].rolling(N1).mean()
    data['MA50'] = data['close'].rolling(N2).mean()
    data['DIF'] = data['MA10'] - data['MA50']
    data['AMA'] = data['DIF'].rolling(M).mean()
    
    return data


# ### Bias

# In[8]:


def BIAS(data, N1, N2, N3):
    
    
    
    
    data['BIAS6'] = (data['close'] - data['close'].rolling(N1).mean())/data['close'].rolling(N1).mean() * 100
    data['BIAS12'] = (data['close'] - data['close'].rolling(N2).mean())/data['close'].rolling(N2).mean() * 100
    data['BIAS24'] = (data['close'] - data['close'].rolling(N3).mean())/data['close'].rolling(N3).mean() * 100
    
    return data


# In[9]:


hq = pd.read_csv('hq.csv',index_col=0)


# In[10]:


hq['date'] = hq['date'].apply(lambda x:x[:4]+x[5:7]+x[8:10])


# In[11]:


hq.head()


# In[12]:


hq.columns


# In[13]:


codelis = hq.code.unique().tolist()


# In[21]:


caiwu = pd.read_csv('caiwu_1.csv',index_col=0)


# In[17]:


caiwu['date'] = caiwu['date'].apply(lambda x:x[:4]+x[5:7]+x[8:10])


# In[18]:


caiwu.head()


# In[20]:


df['month'] = df['date'].apply(str).apply(lambda x:x[:6])


# In[28]:


caiwu['month'] = caiwu['date'].apply(str).apply(lambda x:x[:6])


# In[28]:


all_df = pd.merge(df,caiwu,on=['code','month'],how='left')


# In[31]:


all_df.drop_duplicates(subset=['code','date'],keep='first',inplace = True)


# In[33]:


for i in all_df.columns[1:]:
    all_df[i] = all_df.groupby('code')[i].bfill().ffill()


# In[34]:


all_df


# ### Sector Data

# In[35]:



# from WindPy import w
# w.start()


# In[36]:


# industry_sw = dict()
# for c in tqdm(com_lis):
#     industry_sw[c] = w.wss(c,"industry_sw_2021","tradeDate=20220311;industryType=1").Data[0][0]


# In[38]:


# industry_sw = pd.DataFrame(industry_sw, index = ['industry']).T
# industry_sw.reset_index(inplace = True)
# industry_sw.rename(columns = {'index':'ts_code'},inplace = True)


# In[39]:


# industry_sw


# In[40]:


# industry_sw.to_csv('hy_sw.csv', encoding="gbk") # encoding="gbk" 
# # industry_classify.to_csv('hy_sw.csv', encoding="gbk",index=False)  


# In[41]:


industry_sw = pd.read_csv('hy_sw.csv', encoding="gbk", index_col = 0)


# In[43]:


# industry_sw


# In[44]:


df = pd.merge(all_df,industry_sw,on='code',how='inner')


# In[45]:


df


# In[48]:


# df.to_csv('pre_hy_qx_data_all.csv',encoding = 'gbk')


# ### Building sector factors

# In[47]:



amount_hy = df.groupby(['industry','date'],as_index=False)['amount'].sum()


# In[49]:



amount_hy.columns = ['industry','date','amount_hy']


# In[51]:



df = pd.merge(df,amount_hy,on=['industry','trade_date'],how='inner')


# In[52]:



df['s_m_size'] = df['s_m']*df['amount']/df['amount_hy']
df['close_size'] = df['close']*df['amount']/df['amount_hy']
df['dt_eps_size'] = df['dt_eps']*df['amount']/df['amount_hy']
df['roe_size'] = df['roe']*df['amount']/df['amount_hy']
df['current_ratio_size'] = df['current_ratio']*df['amount']/df['amount_hy']


# In[53]:



industry_fea = df.groupby(['industry','trade_date'],as_index=False)[['s_m_size',
                                                                     'close_size',
                                                                     'dt_eps_size',
                                                                     'roe_size',
                                                                     'current_ratio_size',
                                                                     'amount_hy']].mean()


# In[54]:


# industry_fea


# In[55]:



industry_fea.columns = ['industry',
                        'trade_date',
                        'industry_mom',
                        'industry_close',
                        'industry_PE',
                        'industry_roe',
                        'industry_leverage',
                        'industry_size']


# In[56]:



df = pd.merge(df,industry_fea,on=['industry','trade_date'],how='inner')


# In[57]:



df = df.drop(['s_m_size',
              'close_size',
              'dt_eps_size',
              'roe_size',
              'current_ratio_size',
              'amount_hy'],axis=1) 


# In[61]:


df['trade_date'] = df['trade_date'].astype(str)


# ## Emotional factors

# In[63]:


psy = pd.read_csv('data.csv',index_col=0)


# In[64]:


psy.columns


# In[ ]:



plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# In[65]:


psy.rename(columns={'5年平均收益市值比':'ETP5',
                    '5年平均资产回报率':'ROA5',
                    '5年平均权益回报率':'ROE5',
                    '20日特诺雷比率':'TreynorRatio20',
                    '相对离散指数':'RVI',
                    '阿隆指标':'Aroon',
                    '股价偏度':'Skewness',
                    '5日成交量EMA（手）':'VEMA5',
                    '26日成交量EMA（手）':'VEMA26',
                    '5日平均换手率':'VOL5',
                    '20日平均换手率':'VOL20',
                    '120日平均换手率':'VOL120'},inplace=True)


# In[66]:


psy['trade_date'] = psy['date'].apply(lambda x:x[:4]+x[5:7]+x[8:10])


# In[67]:


psy['ts_code'] = psy['code']


# In[68]:



df = pd.merge(df,psy[['ts_code',
                      'trade_date',
                      'ETP5',
                      'ROA5',
                      'ROE5', 
                      'AR',
                      'BR', 
                      'PSY',
                      'TreynorRatio20',
                      'RVI',
                      'Aroon',
                      'Skewness',
                      'VEMA5',
                      'VEMA26',
                      'VOL5',
                      'VOL20',
                      'VOL120']],on=['ts_code','trade_date'],how='left')


# In[69]:


df['ARBR'] = df['AR'] - df['BR']


# In[73]:


df


# In[74]:


df.columns


# In[75]:


# df.columns[0]


# In[76]:



for i in df.columns[1:]:
    df[i] = df.groupby('ts_code')[i].bfill().ffill()


# In[78]:


df = df.dropna()


# In[ ]:


# df_describe = df.describe().T


# In[ ]:


# df_describe


# In[ ]:


# df_describe.index


# In[ ]:



# df_describe = df_describe.drop(['mark','end_date'])


# In[ ]:



# df_describe.to_csv('factor_describe_raw.csv')


# In[ ]:


# df


# In[85]:


df_linear = df.copy()


# In[86]:


fea_names = [i for i in df_linear.columns if i not in ['industry','ts_code','trade_date','mark','end_date','month','pct_chg','ret','next_ret']]


# In[87]:



# def winsorize(factor,p,data):  
#     data = data.sort_values(factor)    #按factor排序 从小到大
#     data['rank'] = range(len(data))    
#     #down_limit,upper_limit = round(len(data)*p),round(len(data)*(1-p))  #去极值处理的分位点
#     data.loc[:round(len(data)*p),factor] = list(data[factor])[round(len(data)*p)]  #winsorize是用相应分位数的值替代分位数之外的值，而不是删掉
#     data.loc[round(len(data)*(1-p)):,factor] = list(data[factor])[round(len(data)*(1-p))]
#     return data


# In[88]:


for fea in tqdm(fea_names):
    # print(fea,df[fea].mean(),df[fea].std())
    df_linear[fea] = np.where(df_linear[fea]>df_linear[fea].mean()+3*df_linear[fea].std(),df_linear[fea].mean()+3*df_linear[fea].std(), df_linear[fea])
    df_linear[fea] = np.where(df_linear[fea]<df_linear[fea].mean()-3*df_linear[fea].std(),df_linear[fea].mean()-3*df_linear[fea].std(), df_linear[fea])
#     df = winsorize(fea,0.01,df)


# In[91]:


# df_describe = df_linear.describe().T


# In[92]:


# df_describe.index


# In[93]:



# df_describe = df_describe.drop(['mark','end_date'])

# df_describe.to_csv('factor_describe_qujizhihou.csv')


# In[94]:


# # df.info()


# In[95]:


fea_names = [i for i in df_linear.columns if i not in ['industry','ts_code','trade_date','mark','end_date','month','rank','pct_chg','ret','next_ret']]


# In[96]:


for fea in fea_names:
    # print(fea,df[fea].mean(),df[fea].std())
    df_linear[fea] = (df_linear[fea]-df_linear[fea].mean())/df_linear[fea].std()


# In[97]:



# df_describe = df.describe().T

# df_describe = df_describe.drop(['mark','end_date'])

# df_describe.to_csv('factor_describe_final.csv')


# In[98]:



# df_linear.to_csv('standard_data_all.csv',encoding = 'gbk')


# In[102]:



idx_lis = df_linear.industry.unique().tolist()


# In[103]:


d1 = df_linear.set_index('trade_date')


# In[104]:



def test_ic(code,df,rank=False):
    
    df1 = df[df.industry==code] 
    df1.index = pd.to_datetime(df1.index.astype(str))
    df1['next_ret'] = df1.groupby('ts_code')['pct_chg'].shift(-1)
    df2 = df1.resample('W').mean()
    factor = [i for i in df2.columns if i not in ['pct_chg','ret','next_ret','mark','end_date','month']]
    df2 = df2.bfill().ffill().fillna(0)    
    ic_dic = {}
    for fac in factor:
        if rank:
            ic_dic[fac] = df2[fac].rank().corr(df2['next_ret'].rank())
        else:
            ic_dic[fac] = df2[fac].corr(df2['next_ret'])
            
    return ic_dic


# In[105]:


stocklis = d1.ts_code.unique().tolist()


# In[106]:



def load_allstock_factor(stocklis,df):
    all_ic_df = pd.DataFrame()
    for code in tqdm(stocklis):
        dic = test_ic(code,df)
        single_df = pd.DataFrame(dic,index=[code]).T
        all_ic_df = pd.concat([all_ic_df,single_df],axis=1)
    return all_ic_df


# In[107]:



ic_val = load_allstock_factor(idx_lis,d1)


# In[108]:



ic_val


# In[109]:



sig = abs(ic_val.mean(1)).dropna()


# In[110]:


sig.sort_values(ascending=False)


# In[203]:



industry_IC = sig.sort_values(ascending=False)
# industry_IC.to_csv('industry_IC.csv')


# In[204]:


industry_IC


# In[113]:


# len(industry_IC)


# In[114]:



feanames = [i for i in df_linear.columns if i not in ['industry','ts_code','trade_date','pct_chg','ret','next_ret','mark','end_date']]


# In[115]:



def test_ic_TimeSeries(factor,df,rank=False,num=20):
    print('Factor:',factor)
    df1 = df[[factor,'next_ret','trade_date']]
    # df1.index = pd.to_datetime(df1.index.astype(str))
    # df2 = df1.resample('W').mean()
    ic_dic = {}
    ic_df = df1.groupby('trade_date')[[factor,'next_ret']].mean()
    for i in range(num, ic_df.shape[0],num):
        test = ic_df.iloc[i-num:i]
        ic_dic[i-num] = test[factor].corr(test['next_ret'])
    ic_plot = pd.DataFrame(ic_dic,index=['cor']).T
    # ic_plot.index = df1.trade_date.unique()[num:]
    # ic_plot.index = pd.to_datetime(ic_plot.index)
    df = ic_plot.reset_index()
    # df.columns = ['trade_date', ids]
    df.set_index('index')['cor'].plot(figsize=(12,8),grid=True,title='IC Time Series Plot')

    model = sm.OLS(df1['next_ret'].values,df1[factor].values.reshape(-1,1))
    results = model.fit()
    print(results.summary())
    return 


# In[116]:


test_ic_TimeSeries('dt_eps',df_linear) #industry_mom


# In[2]:


from scipy import stats, optimize


# In[ ]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(d1[feanames[0]])
plot_pacf(d1[feanames[0]])


# In[ ]:





# In[117]:


def multifactor_for_stock(code,df):
    
    dt = df[df.ts_code == code]
    X_train = dt[feanames].values
    Y_train = dt['next_ret'].values
    model = LinearRegression()
    model.fit(X_train,Y_train)
    a  = model.intercept_
    b = model.coef_
    print("股票alpha:",a)
    tb = pd.DataFrame(b,index=feanames,columns=['factor_ret']) 
    print(tb)
    
    return tb


# In[125]:


tb = multifactor_for_stock(stocklis[0],df_linear)


# In[126]:


tb.sort_values(by = 'factor_ret',ascending = False,inplace = True)


# In[129]:


choose_fac = sig.sort_values(ascending=False).index.tolist()[:60]


# In[130]:



fea_names = choose_fac


# In[131]:



corr_df = df_linear[fea_names].corr()


# In[2]:


# Train:2018-01-01，Test:2018-01-01


# In[132]:



d1['mark'] = np.sign(d1.groupby('ts_code')['pct_chg'].shift(-1))


# In[133]:


df_ols = d1.reset_index()
df_ols['date'] = df_ols['trade_date'].apply(str).apply(lambda x:x[:10])


# In[134]:


df_ols = df_ols.dropna()


# In[135]:



df_ols = df_ols.dropna()
train_data_ols = df_ols[df_ols['date'] < time]
test_data_ols = df_ols[df_ols['date'] >= time]
# cost = 0.0003
X_train_ols = train_data_ols[fea_names]
X_test_ols = test_data_ols[fea_names]
y_train_ols = train_data_ols['mark']
y_test_ols = test_data_ols['mark']
test_data_ols['pct_chg'] = test_data_ols['pct_chg']/100


# In[141]:



df['mark'] = np.sign(df.groupby('ts_code')['pct_chg'].shift(-1))


# In[142]:


df['date'] = df['trade_date'].apply(str).apply(lambda x:x[:10])


# In[143]:


df = df.dropna()


# In[144]:



time = "20180101"
df = df.dropna()
train_data = df[df['date'] < time]
test_data = df[df['date'] >= time]
# cost = 0.0003
X_train = train_data[fea_names]
X_test = test_data[fea_names]
y_train = train_data['mark']
y_test = test_data['mark']
test_data['pct_chg'] = test_data['pct_chg']/100


# ### The prediction accuracy of the model was calculated

# In[145]:


def model_evaluation(estimator, train_data, real_mark):
    predict = estimator.predict(train_data)
    compare_df = pd.DataFrame({'Real': real_mark, 'Predict': predict})
    compare_df['mark'] = ~np.logical_xor(compare_df.Predict, compare_df.Real)
    score = estimator.score(train_data, real_mark)
    print(f'Accuracy: {score}\n\nConfusion matrix:\n')
    confusion = confusion_matrix(real_mark, predict, labels=[0,1])
    confusion_df = pd.DataFrame(confusion, index=['0_real', '1_real'], columns=['0_predict', '1_predict'] )
    return confusion_df


# In[146]:


clf_ols = LogisticRegression()
clf_ols.fit(X_train_ols, y_train_ols)
model_evaluation(clf_ols, X_train_ols, y_train_ols)


# In[147]:


clf_forest = RandomForestClassifier(n_estimators=10, max_features='sqrt', max_depth=100, class_weight='balanced')
clf_forest.fit(X_train,y_train)
model_evaluation(clf_forest, X_train, y_train)


# In[148]:


model_evaluation(clf_forest, X_test, y_test)


# In[149]:


df['mark_xg'] = df['mark'].replace(-1,0)


# In[150]:


y_train_xg = y_train.replace(-1,0)


# In[151]:


y_test_xg = y_test.replace(-1,0)


# In[152]:


data = df[fea_names] 


label = df['mark_xg']
dtrain = xgb.DMatrix(X_train, label = y_train_xg)
dtest = xgb.DMatrix(X_test)


params={'booster':'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth':4,
    'lambda':10,
    'subsample':0.75,
    'colsample_bytree':0.75,
    'min_child_weight':2,
    'eta': 0.025,
    'seed':0,
    'nthread':8,
     'silent':1}

watchlist = [(dtrain,'train')]

bst=xgb.train(params,dtrain,num_boost_round=100,evals=watchlist)
ypred=bst.predict(dtest)


y_pred = (ypred >= 0.3)*1


print ('AUC: %.4f' % metrics.roc_auc_score(y_test_xg,ypred))


print ('ACC: %.4f' % metrics.accuracy_score(y_test_xg,y_pred))
print ('Recall: %.4f' % metrics.recall_score(y_test_xg,y_pred))


print ('F1-score: %.4f' %metrics.f1_score(y_test_xg,y_pred))
print ('Precesion: %.4f' %metrics.precision_score(y_test_xg,y_pred))
confusion_matrix(y_test_xg,y_pred)


# # The stock returns are predicted based on the model

# In[156]:


predict = clf_forest.predict(X_test)
test_data['predict'] = predict

# Take a full position on bears and a full position on bulls
test_data['predict'] = np.where(test_data['predict'] <= 0 ,0.5,1)
test_data['rets'] = test_data['predict'].shift(1) * test_data['pct_chg']


# In[157]:


test_data['predict_xg'] = y_pred
test_data['predict_xg'] = np.where(test_data['predict_xg'] <= 0 ,0.5,1)
test_data['rets_xg'] = test_data['predict_xg'].shift(1) * test_data['pct_chg']


# In[158]:


predict_ols = clf_ols.predict(X_test_ols)
test_data['predict_ols'] = predict_ols
test_data['predict_ols'] = np.where(test_data['predict_ols'] <= 0 ,0.5,1)
test_data['rets_ols'] = test_data['predict_ols'].shift(1) * test_data['pct_chg']


# In[160]:


test_data = test_data.dropna()


# In[161]:


test_data['trade_date'] = test_data['trade_date'].astype(str)


# In[162]:


ret_df = test_data.groupby('trade_date')[['rets_xg','rets','pct_chg','rets_ols']].mean()
# ret_df = test_data.groupby('trade_date')[['rets_xg','rets','pct_chg','rets_ols','rets_NB']].mean()
ret_df['benchmark'] = ret_df['pct_chg']
ret_df['OLS'] = ret_df['rets_ols'] + ret_df['pct_chg']
ret_df['RF'] = ret_df['rets'] + ret_df['pct_chg']
ret_df['XGBoost'] = ret_df['rets_xg'] + ret_df['pct_chg']
# ret_df['NB'] = ret_df['rets_NB'] + ret_df['pct_chg']


# In[163]:


ret_df.index = pd.to_datetime(ret_df.index)


# In[164]:


ret_df[['OLS']].cumsum().plot(figsize=(12,8),grid=True)


# # HS300

# In[1]:


#From tushare


# In[166]:


hs300 = pd.read_csv('hs300.csv',index_col=0)
# hs300


# In[167]:


ret_df['HS300'] = hs300['pct_chg'].values/100


# In[168]:


ret_df[['OLS','HS300']].cumsum().plot(figsize=(12,8),grid=True)


# # set initial capital M

# In[29]:


M = 10000000


# In[ ]:


money_df = M*ret_df


# In[169]:


ret_df[['XGBoost','RF','HS300']].cumsum().plot(figsize=(12,8),grid=True)


# In[170]:


ret_df[['XGBoost','RF','OLS','HS300']].cumsum().plot(figsize=(12,8),grid=True)


# In[ ]:


money_df[['XGBoost','RF','HS300']].cumsum().plot(figsize=(12,8),grid=True)


# In[171]:


def ratio(data):
    
    strategy_cum = (data + 1).cumprod()
    return_year = data.mean() * 252 
    volatility = data.std() * 252 ** 0.5   #volatility = df['strategy'].std()/ (np.sqrt(1/252) )
    profit_max = data.max()
    loss_max = data.min()
    sharpe = (return_year - 0.02) / volatility 
    IC_ratio = (return_year - 0.03) / volatility
    
    tmp = pd.DataFrame([ float((strategy_cum).tail(1)), sharpe, return_year, volatility, profit_max, loss_max, IC_ratio], 
                       columns=[data.name], 
                       index=[ 'Total Return', 'Sharpe Ratio' , 'CAGR', 'Annualised Volatility', 'Maximum Daily Profit', 
                              'Maximum Daily Loss','IC Ratio'])
    tmp.columns = ['Performance']
#         res.append(tmp)

    return tmp


# In[176]:


# ratio(ret_df['RF'])


# In[177]:


# ratio(ret_df['HS300'])


# In[178]:


# ratio(ret_df['XGBoost'])


# In[179]:


# ratio(ret_df['OLS'])


# In[180]:



look_XGBoost = ratio(ret_df['XGBoost'])
look_RF = ratio(ret_df['RF'])
look_OLS = ratio(ret_df['OLS'])
look_hs300 = ratio(ret_df['HS300'])
look_all = pd.concat((look_XGBoost,look_RF,look_OLS,look_hs300),axis = 1)


# In[181]:



look_all.columns = ['XGBoost','RF','OLS','HS300']


# In[182]:


look_all


# In[183]:


num = 20
importances = clf_forest.feature_importances_ 
importances_df = pd.DataFrame([fea_names, importances], index=['Features', 'Importances']).T
importances_df.sort_values(by='Importances',ascending=False).head(num)


# In[184]:


importances_df.set_index('Features').sort_values(by='Importances',ascending=False).head(num).plot.bar(figsize=(14,6),title='the importance of factors')


# In[185]:


ip_lis = importances_df.set_index('Features').sort_values(by='Importances',ascending=False).head(num).index.tolist()


# In[186]:


X_train[ip_lis].corr()


# In[187]:


def dataPlot():

    fig,ax=plt.subplots(figsize=(18,8))
    sns.heatmap(pd.DataFrame(np.round(X_train[ip_lis].corr(),4),columns=ip_lis,index=ip_lis),
                annot=True,
                vmax=1,
                vmin=0,
                xticklabels=True,
                yticklabels=True,
                square=True,
                cmap="YlGnBu")
    ax.set_title(' the importance of factors ', fontsize=18)
    ax.set_ylabel('Y', fontsize=16)
    ax.set_xlabel('X', fontsize=16)


# In[188]:


dataPlot()


# In[189]:


importances_xg = bst.get_fscore()
importances_xg_df = pd.DataFrame([importances_xg], index=['Importances']).T
importances_xg_df.reset_index(inplace = True)
importances_xg_df.rename(columns={'index':'Features'},inplace=True)
importances_xg_df.sort_values(by='Importances',ascending=False).head(num)


# In[190]:



importances_xg_df.set_index('Features').sort_values(by='Importances',ascending=False).head(num).plot.bar(figsize=(14,6),title='the importance of factors')


# In[191]:


ip_lis_xg = importances_xg_df.set_index('Features').sort_values(by='Importances',ascending=False).head(num).index.tolist()


# In[192]:


X_train[ip_lis_xg].corr()


# In[193]:


fig,ax=plt.subplots(figsize=(18,8))
sns.heatmap(pd.DataFrame(np.round(X_train[ip_lis_xg].corr(),4),columns=ip_lis_xg,index=ip_lis_xg),
            annot=True,
            vmax=1,
            vmin=0,
            xticklabels=True,
            yticklabels=True,
            square=True,
            cmap="YlGnBu")
ax.set_title(' the importance of factors ', fontsize=18)
ax.set_ylabel('Y', fontsize=16)
ax.set_xlabel('X', fontsize=16)


# In[194]:


importances_ols = clf_ols.coef_[0]


# In[195]:


importances_ols_df = pd.DataFrame(importances_ols,index = fea_names)
importances_ols_df.reset_index(inplace = True)
importances_ols_df.rename(columns={'index':'Features',0:'Importances'},inplace=True)


# In[196]:


importances_ols_df.sort_values(by='Importances',ascending=False).head(20)


# In[197]:


ip_lis_ols = importances_ols_df.set_index('Features').sort_values(by='Importances',ascending=False).head(num).index.tolist()


# In[198]:


fig,ax=plt.subplots(figsize=(18,8))

sns.heatmap(pd.DataFrame(np.round(X_train[ip_lis_ols].corr(),4),columns=ip_lis_ols,index=ip_lis_ols),
            annot=True,
            vmax=1,
            vmin=0,
            xticklabels=True,
            yticklabels=True,
            square=True,
            cmap="YlGnBu")
ax.set_title(' the importance of factors ', fontsize=18)
ax.set_ylabel('Y', fontsize=16)
ax.set_xlabel('X', fontsize=16)


# In[199]:


# from sklearn.inspection import permutation_importance
# imps = permutation_importance(NB, X_test, y_test)
# importances_NB = imps.importances_mean


# In[200]:


# importances_NB_df = pd.DataFrame(importances_NB,index = fea_names)
# importances_NB_df.reset_index(inplace = True)
# importances_NB_df.rename(columns={'index':'Features',0:'Importances'},inplace=True)


# In[201]:


# importances_NB_df.sort_values(by='Importances',ascending=False).head(num)


# In[202]:


# ip_lis_NB = importances_ols_df.set_index('Features').sort_values(by='Importances',ascending=False).head(num).index.tolist()
# fig,ax=plt.subplots(figsize=(18,8))
# sns.heatmap(pd.DataFrame(np.round(X_train[ip_lis_NB].corr(),4),columns=ip_lis_NB,index=ip_lis_NB),
#             annot=True,
#             vmax=1,
#             vmin=0,
#             xticklabels=True,
#             yticklabels=True,
#             square=True,
#             cmap="YlGnBu")
# ax.set_title(' the importance of factors ', fontsize=18)
# ax.set_ylabel('Y', fontsize=16)
# ax.set_xlabel('X', fontsize=16)


# ## Important factor

# In[ ]:


# ip_lis
# ip_lis_xg
# ip_lis_ols
# ip_lis_NB


# In[205]:



ip_lis_ic = industry_IC.head(20).index.tolist()


# In[207]:



all_fea = ip_lis+ip_lis_xg+ip_lis_ols+ip_lis_ic
# all_fea = ip_lis+ip_lis_xg+ip_lis_ols+ip_lis_NB+ip_lis_ic


# In[208]:


len(all_fea)


# In[209]:



summ = {}
for i in set(all_fea):
    summ[i] = all_fea.count(i)


# In[210]:



len(summ)


# In[211]:



summ = pd.DataFrame(summ,index = ['count']).T.sort_values('count',ascending=False)


# In[212]:


summ


# In[213]:



summ_2 = summ[summ['count']>1].index.tolist()


# In[215]:



len(summ_2)


# Among the top 30 important factors, industry_size, industry_LEVERAGE and Industry_MOM are constructed in this paper

# In[216]:



summ_2


# In[54]:


from sklearn.ensemble import RandomForestClassifier
clf_forest = RandomForestClassifier(n_estimators=10, max_features='sqrt', max_depth=100, class_weight='balanced')
hs300 = pro.index_daily(ts_code='399300.SZ', start_date='20100101', end_date='20211231').sort_values(by='trade_date')


# In[123]:


# fea_names = ['close', 'open', 'high', 'low', 'pre_close','change','vol','amount']
# hs300['ret'] = hs300['pct_chg']/100
hs300['mark'] = np.where(hs300['ret'].shift(-1)>0.02,1,np.nan)
# hs300['mark'] = np.where(hs300['ret'].shift(-1)<0,-1,hs300['mark'])
hs300['mark'] = hs300['mark'].fillna(0)


# # Import the fundamental factors of csi 300 index data, combined with market factors to predict the trend of the index, as a signal to adjust positions

# In[124]:


idx_factor = pd.read_csv('idx_factor.csv',index_col=0)


# In[125]:


hs300


# In[126]:


idx_factor['trade_date'] = idx_factor['date'].apply(lambda x:x[:4]+x[5:7]+x[8:10])
# idx_factor['ts_code'] = idx_factor['code']


# In[127]:


idx_factor = idx_factor.drop(['code','date'],axis=1)


# In[128]:


hs300 = pd.merge(hs300,idx_factor,on=['trade_date'],how='left')


# In[129]:


time = '20180101'
train_data = hs300[hs300['trade_date'] < time]
test_data = hs300[hs300['trade_date'] >= time]


# In[130]:


hs300['mark'].unique()


# In[131]:


hs300.columns


# In[132]:


fea_names = [i for i in hs300.columns if i not in ['ts_code','mark']]


# In[133]:


# Split the training set and test set according to time, with training set prior to 2018
X_train = train_data[fea_names]
X_test = test_data[fea_names]
y_train = train_data['mark']
y_test = test_data['mark']


# In[134]:


clf_forest.fit(X_train,y_train)


# In[135]:


predict = clf_forest.predict(X_test)
test_data['predict'] = predict


# In[141]:


map_dic = {-1:0,0:1.5,1:1}


# In[142]:


test_data['lot'] = test_data['predict'].map(map_dic)


# In[143]:


test_data['strg_ret'] = test_data['lot'] * test_data['ret']


# In[144]:


test_data.set_index('trade_date')[['strg_ret','ret']].cumsum().plot(figsize=(18,8),grid=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[20]:


hs300 = pro.index_daily(ts_code='000050.SH', start_date='20210526', end_date='20211231').sort_values(by='trade_date')
hs300['pct_chg'] = hs300['pct_chg']/100


# In[21]:


(hs300['close'].values[-1]/hs300['close'].values[0]) - 1


# In[15]:


hs300


# In[ ]:




