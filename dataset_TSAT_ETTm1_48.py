from ast import parse
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

from PyEMD import EMD,CEEMDAN
import seaborn as sns
import matplotlib.pyplot as plt
# pip install EMD-signal


def emd_imf(signal):                               # IMFs是一组具有不同频率和幅度的信号，可以通过叠加这些IMFs来重构原始信号。
    """
    This function is to calculate EMD of the time series.
    :params: signal: list

    :return: res_dict: a dict consists of the different imf list value
    """
    if isinstance(signal, list):
        signal = np.array(signal)
    assert isinstance(signal, np.ndarray)              # 确保其为numpy类型
    IMFs = EMD().emd(signal, np.arange(len(signal)))   # 使用EMD算法对输入信号进行经验模态分解，得到一组内禀模态函数（IMFs）
    # IMFs = CEEMDAN().emd(signal, np.arange(len(signal)))
    res_dict = {}
    for _ in range(IMFs.shape[0]):
        res_dict[f'imf_{_}'] = IMFs[_].tolist()
    return res_dict


def calculate_imf_features(n_zones, index_pair_for_one, zones_dict:dict, ticker_SMD_dict:dict, n_imf_use=5) -> np.ndarray:
    """
    compute the EMD.

    :return: imf_features
    """
    assert isinstance(n_imf_use, int)
    imf_features = np.zeros((n_zones, n_zones, n_imf_use))       # 创建一个n_imf_use=5的全零矩阵 7*7*5
    ticker_A, ticker_B = None, None       # 用于跟踪当前处理的节点名称
    for pair in index_pair_for_one:       # pair索引对
        #  如果当前处理节点的名称（通过pair[0]索引到的节点）与ticker_A不相同，则更新ticker_A为当前节点，并从ticker_SMD_dict中获取该节点的经验模态分解结果，存储在变量ticker_A_SMD中。
        if ticker_A != zones_dict[pair[0]]:
            ticker_A = zones_dict[pair[0]]
            ticker_A_SMD = ticker_SMD_dict[ticker_A]
        if ticker_B != zones_dict[pair[1]]:
            ticker_B = zones_dict[pair[1]]
            ticker_B_SMD = ticker_SMD_dict[ticker_B]
        
        ef = [0] * n_imf_use     # 创建一个长度为n_imf_use的列表ef，并将其中所有元素初始化为0。
        for n_imf in list(range(1, n_imf_use+1)):  # n_imf_to_exact = n_imf_use   1-6直接取了中间的5个，但是前面为啥提取8个，那我直接取5个不就行啦
            if f'imf_{n_imf}' in ticker_A_SMD and f'imf_{n_imf}' in ticker_B_SMD:
                # to get both imf for both 2 tickers
                ef[n_imf-1] = (np.corrcoef(ticker_A_SMD[f'imf_{n_imf}'],          # 使用np.corrcoef计算两个IMF的相关系数
                                            ticker_B_SMD[f'imf_{n_imf}'])[0][1]   # 将相关系数值存储在ef列表中的第n_imf-1个位置（由于列表索引从0开始）。
                                )
            else:  # exit the loop when there is no further imf correctlation
                break          # 不是一对imf，因此没有东西可以算相关性    ticker_A_SMD或ticker_B_SMD字典中不存在名为imf_{n_imf}的键，则退出循环
        imf_features[pair[0]][pair[1]], imf_features[pair[1]][pair[0]] = np.array(ef), np.array(ef)
    
    return imf_features


def process_data(df, n_lookback_days, n_lookforward_days, adj_mat_method='fully_connected', use_tqdm=True, **kwargs):
    """
    This is the main part of generate graph.
    """
    if n_lookforward_days > n_lookback_days:
        warnings.warn(f'The number of lookforward days ({n_lookforward_days}) is lager than lookback days ({n_lookback_days}). Please conside using longer lookback days')
    _graph_data, _graph_label = [], []
    zones = df.columns.to_list()   # 每一列的列名存在列表里
    n_zones, zones_dict = len(zones), dict(zip(range(len(zones)), zones))     # 228*228矩阵，zones_dict分别用0123456对应列名
    # nf_global_max, nf_global_min = df.max().to_numpy(), df.min().to_numpy()
    ranges = range(len(df) - n_lookforward_days - n_lookback_days)   # 包含0，range（0, 3787）
    print(f'tqdm used: {use_tqdm}')
    range_iterable = tqdm(ranges) if use_tqdm else ranges
    for adate in range_iterable:    # 挨个取样本
        if not use_tqdm:
            print(f'Generating graph for time {df.index[adate+n_lookback_days]} ...... ', end='')
        
        lookback_period_df = df[adate:adate+n_lookback_days]    # 初始点adate值为0，x选取0-11   df(12, 228)
        lookforward_period_df = df[adate+n_lookback_days:adate+n_lookback_days+n_lookforward_days]  # y为11+1  df(1, 228)
    
        # node_features  每个节点在过去所有历史时刻构成的列表
        node_features = lookback_period_df.to_numpy().transpose()   # 每个节点在过去T个历史时刻flow构成的列表
        
        # node_features (normalization)
        nf_max, nf_min = np.amax(node_features, axis=1), np.amin(node_features, axis=1)    # 在axis=1维度上进行节点特征的最小值最大值计算
        # if encounter constant series, i.e., max = min. The result will be all zero.
        if np.any(nf_max == nf_min):   # 判断是否是常数列
            nf_max[np.where(nf_max - nf_min == 0)], nf_min[np.where(nf_max - nf_min == 0)] = 1, 0
        
        nf_MIN = np.repeat(nf_min, n_lookback_days, axis=0).reshape(n_zones, n_lookback_days)   # 将最小值复制n_lookback_days（720）次，重塑为形状为 (n_zones, n_lookback_days) 的矩阵
        nf_MAX = np.repeat(nf_max, n_lookback_days, axis=0).reshape(n_zones, n_lookback_days)
        node_features_normalization = (node_features-nf_MIN)/(nf_MAX - nf_MIN)
        
        # adj_matrix (fully connected or corr)
        if adj_mat_method == 'fully_connected':    # 全连接，就是除对角线其余对角线上的元素全为0，所有节点之间都存在链接
            # all 1 except the diagonal
            adj_mat = np.ones((n_zones, n_zones)) - np.eye(n_zones)
        elif adj_mat_method == 'correlation':      # 皮尔逊相关
            # based on correlation
            correlation_matrix = np.abs(np.corrcoef(lookback_period_df.values, rowvar=False))    # 计算给定数据lookback_period_df的相关系数矩阵，并把对角线元素设为0
            correlation_matrix = np.where(correlation_matrix == 1., 0, correlation_matrix)
            correlation_matrix = np.where(correlation_matrix >= 0.8, 1, 0) # 0.75  根据设定的阈值（此处为0.8），将相关系数大于等于阈值的元素设置为1，表示节点之间存在连接，否则设置为0。
            adj_mat = correlation_matrix
        elif adj_mat_method == 'zero_mat':
            # zero matrix
            adj_mat = np.zeros((n_zones, n_zones))      # 生成一个全为零的邻接矩阵，表示没有节点之间的连接。
        elif adj_mat_method == 'random':
            # random
            b = np.random.random_integers(0, 1, size=(n_zones, n_zones))  # 生成一个元素为0或1的随机矩阵
            adj_mat = b * b.T                                             # 生成的矩阵与其转置相乘确保对称性
        else:
            raise TypeError(f'Unsupported adj_matrix method: {adj_mat_method}!')    # 不是以上矩阵，抛出异常
        
        ## calculate_imf_features
        # 在adj的上三角找索引对
        index_pair_for_one = np.argwhere(np.triu(adj_mat) == 1)   # 使用np.triu()函数获取邻接矩阵的上三角部分，使用np.argwhere()函数找到值为1的元素的索引对（表示有连接关系）
        ticker_SMD_dict = dict.fromkeys(zones)      # 创建一个字典，内容为228个节点的索引
        involde_index_idxs_np = np.unique(index_pair_for_one.flatten())    # 将索引对展平去重，得到涉及到的索引值，这些索引值对应于具有连接关系的节点
        for index_idx in involde_index_idxs_np:
            ticker = zones_dict[index_idx]       # 在zone_dict根据索引获取相应节点
            ticker_SMD_dict[ticker] = emd_imf(lookback_period_df[ticker].to_list())    # 将相应节点的时间序列数据lookback_period_df[ticker].to_list()）作为参数进行经验模态分解（EMD），并将结果存储在ticker_SMD_dict[ticker]中。emd分解，分解为8个

        #   7个节点 ，每个特征有个imfs，都存在ticker_SMD_dict
        imf_features = calculate_imf_features(n_zones, index_pair_for_one, zones_dict, ticker_SMD_dict, n_imf_use=5)

        # 将IMF特征矩阵以对应的键名存储在imf_matries_dict字典中  5个7*7矩阵
        imf_matries_dict = {
            'imf_1_matix':imf_features[:,:,0],    # 表示IMF特征的第一个矩阵。
            'imf_2_matix':imf_features[:,:,1],
            'imf_3_matix':imf_features[:,:,2],
            'imf_4_matix':imf_features[:,:,3],
            'imf_5_matix':imf_features[:,:,4],
        }
        
        # label
        label = lookforward_period_df.to_numpy().transpose()

        # label (normalization)
        label_MIN = np.repeat(nf_min, n_lookforward_days, axis=0).reshape(n_zones, n_lookforward_days)
        label_MAX = np.repeat(nf_max, n_lookforward_days, axis=0).reshape(n_zones, n_lookforward_days)
        label_normalization = (label-label_MIN)/(label_MAX - label_MIN)
        label_normalization = label_normalization.flatten()
        
        name_of_sample = f"ETTm1_LookbackInculde[{lookback_period_df.index[0]} to {lookback_period_df.index[-1]}]_Predict[{lookforward_period_df.index[0]} to {lookforward_period_df.index[-1]}]"
        
        
        G = {
            'date': name_of_sample,
            'node_feat': node_features_normalization,
            'imf_matries_dict': imf_matries_dict,   # 228*228*5
            'adj_mat': adj_mat,                   # 皮尔逊系数算的 228*228
            'node_local_MAX': nf_max,
            'node_local_MIN': nf_min,
            'node_global_MAX': None,
            'node_global_MIN': None
        }
        _graph_data.append(G)
        _graph_label.append(list(label_normalization))    # 48*7的矩阵转为list 336的list 即只预测336个值即可
        if not use_tqdm:
            print('Done !')
            
    return _graph_data, _graph_label


def list_split(L, n_split):
    """
    Split the list
    """
    assert isinstance(L, list)
    k, m = divmod(len(L), n_split)
    _ = (L[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_split))
    return list(_)


def generate_ETTm1_data(n_lookback_days:int, n_lookforward_days:int, adj_mat_method:str, use_tqdm:bool):
    import time
    start_time = time.time()
    # df = pd.read_csv(f'Data/ETTm1_{n_lookforward_days}/source/ETTm1.csv', index_col='date')   # 69680*7
    df = pd.read_csv(f'Data/PEMS/PEMS_BAY.csv', header=None)
    assert len(df) != 0                     # check the input df, it should noe be None.
    # df = df[:200]
    # df = df[:5000]   # 16700
    # fill填充
    df.ffill(inplace=True)      # 前向填充
    df.bfill(inplace=True)
    graph_data, graph_label = process_data(df, n_lookback_days, n_lookforward_days, adj_mat_method=adj_mat_method, use_tqdm=use_tqdm)    # correlation
    assert len(graph_data) == len(graph_label)  # the length of data_mol and label should be same
    assert graph_data[0] is not None      # Check the datamol should be non-empty
    end_time = time.time()
    print(f'Generate ETTm1 data complete! Total time: {end_time-start_time}')
    return graph_data, graph_label

# ======================================================================================


if __name__ == '__main__':
    # init args
    parser = argparse.ArgumentParser()
    parser.add_argument("--lb", type=int, help="n_lookback_days", default=12)
    parser.add_argument("--lf", type=int, help="n_lookforward_days", default=12)
    parser.add_argument("--adj_mat_method", type=str, help="adj_mat_method", default='zero_mat', choices=['zero_mat', 'fully_connected', 'correlation', 'random'])
    parser.add_argument("--use_tqdm", type=str, help="use tqdm or not", default='True', choices=['True', 'False'])
    args = parser.parse_args()
    assert isinstance(args.lb, int)
    assert isinstance(args.lf, int)
    # generate ETTm1_48 datas
    graph_data, graph_label = generate_ETTm1_data(args.lb, args.lf, args.adj_mat_method, bool(args.use_tqdm))


