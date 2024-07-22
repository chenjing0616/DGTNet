import argparse
from collections import defaultdict
from dataset_DGTNet_ETTm1_48 import generate_ETTm1_data
from dataset_graph import construct_DGTNet_dataset, graph_collate_func_DGTNet_normalization_require
import numpy as np
from sklearn.model_selection import train_test_split
from DGT import make_DGTNet_model
import time
import torch
from torch.utils.data import DataLoader
from utils import DGTNet_parameter, loss_function, calculate_loss
from sklearn import metrics
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
# torch.backends.cuda.max_memory_split = 128

# 保存模型
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print("模型已保存至:", path)

# 计算RMSE
def calculate_rmse(y_true, y_pred):
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    return rmse
# MAE
def calculate_mae(y_true, y_pred):
    mae = np.mean(np.abs(y_pred - y_true))
    return mae
# MAPE
def calculate_mape(y_true, y_pred):
    absolute_error = np.abs(y_pred - y_true)
    relative_error = absolute_error / (np.abs(y_true) + 1)  # 添加一个小的常数，避免除零错误
    mape = np.mean(relative_error)
    return mape


class KOI_model_train_test_interface():    # 定义的接口
    def __init__(self, DGTNet_model, model_params:dict, train_params:dict) -> None:
        self.DGTNet_model = DGTNet_model
        self.DGTNet_model = self.DGTNet_model.to(train_params['device'])    # send the model to GPU
        self.train_params = train_params
        self.model_params = model_params
        self.criterion = loss_function(train_params['loss_function'])
        # self.optimizer = torch.optim.Adam(self.DGTNet_model.parameters())

    def import_dataset(self, dataset) -> None:
        # import the dataset
        # 划分比列，直接写索引分比较好
        train_valid_split_ratio = 0.2
        num_workers = 0   # 20
        train_valid_dataset, self.test_dataset = train_test_split(dataset, test_size=train_valid_split_ratio, shuffle=True)  # False
        self.train_dataset, self.valid_dataset = train_test_split(train_valid_dataset, test_size=train_valid_split_ratio)
        # Data Loader
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.train_params['batch_size'], 
                                       collate_fn=graph_collate_func_DGTNet_normalization_require, shuffle=True,
                                       drop_last=True, num_workers=num_workers, pin_memory=False)
        
        self.valid_loader = DataLoader(dataset=self.valid_dataset, batch_size=self.train_params['batch_size'], 
                                       collate_fn=graph_collate_func_DGTNet_normalization_require, shuffle=True,
                                       drop_last=True, num_workers=num_workers, pin_memory=False)

    def calculate_number_of_parameter(self) -> int:
        model_parameters = filter(lambda p: p.requires_grad, self.DGTNet_model.parameters())
        total_parameters = sum([np.prod(p.size()) for p in model_parameters])
        print("==========total_parameters==========")
        print("总模型参数量:", total_parameters)
        return int(sum([np.prod(p.size()) for p in model_parameters]))

    def view_train_params(self):
        # view the training parameters
        for key, value in train_params.items():
            print(key, ":", value)
        print("==========train_params==========")
        print("训练参数: ",train_params)
        return self.train_params
    
    def view_model_params(self):
        # view the model parameters
        for key, value in model_params.items():
            print(key, ":", value)
        print("==========model_params==========")
        print("模型参数: ",model_params)
        return self.model_params

    def train_model(self) -> None:
        # training start
        start_time = time.time()
        self.DGTNet_model.train()
        for batch in self.train_loader:
            graph_name_list, adjacency_matrix, node_features, imf_1_matrices, imf_2_matrices, imf_3_matrices, imf_4_matrices, imf_5_matrices, y_true_normalization, _, _ = batch
            adjacency_matrix = adjacency_matrix.to(self.train_params['device'])  # (batch, max_length, max_length)
            node_features = node_features.to(self.train_params['device'])   # (batch, max_length, d_node)
            imf_1_matrices = imf_1_matrices.to(self.train_params['device']) # (batch, max_length, max_length)
            imf_2_matrices = imf_2_matrices.to(self.train_params['device']) # (batch, max_length, max_length)
            imf_3_matrices = imf_3_matrices.to(self.train_params['device']) # (batch, max_length, max_length)
            imf_4_matrices = imf_4_matrices.to(self.train_params['device']) # (batch, max_length, max_length)
            imf_5_matrices = imf_5_matrices.to(self.train_params['device']) # (batch, max_length, max_length)

            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0  # (batch, max_length)     # 标识批次中的节点是否有效
            y_true_normalization = y_true_normalization.to(self.train_params['device'])  # (batch, task_numbers)
            # 在调用 self.DGTNet_model() 之前计算 y_pred_normalization
            y_pred_normalization = self.DGTNet_model(
                node_features, batch_mask, adjacency_matrix, imf_1_matrices, imf_2_matrices, imf_3_matrices,
                imf_4_matrices, imf_5_matrices
            )
        # save model
        end_time = time.time()
        print(f'DGTNet train complete! Training time: {end_time-start_time}')

    def test_model(self) -> None:
        # testing start
        start_time = time.time()
        # load model and no_grad
        end_time = time.time()
        print(f'DGTNet test complete! Testing time: {end_time - start_time}')


if __name__ == '__main__':
    ## init args
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, help='gpu', default=0)
    parser.add_argument("--dataset", type=str, help='name of dataset', default='PEMS_BAY')
    args = parser.parse_args()
    print(args)

    DGTNet_parameters = DGTNet_parameter(args.dataset)    # 添加参数
    model_params, train_params = DGTNet_parameters.parameters()

    ## Check GPU is available
    train_params['device'] = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')
    # train_params['device'] = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # generate graph from data
    data_graph, data_label = generate_ETTm1_data(
        n_lookback_days=12,
        n_lookforward_days=12,
        adj_mat_method='correlation',       # 皮尔逊系数，使用相关性方法来计算节点之间的连接权重
        use_tqdm=True
        )

    # dataset是list 332
    dataset = construct_DGTNet_dataset(data_graph, data_label, normalization_require=True)
    total_metrics = defaultdict(list)

    ## main train and test
    DGTNet_model = make_DGTNet_model(**model_params)
    model_interface = KOI_model_train_test_interface(DGTNet_model, model_params, train_params)      # 一个接口
    model_interface.import_dataset(dataset=dataset)
    model_interface.train_model()
    model_interface.test_model()
    model_interface.calculate_number_of_parameter()
    model_interface.view_model_params()
    model_interface.view_train_params()


    # 测试数据集
    num_workers = 0
    test_loader = DataLoader(dataset=model_interface.test_dataset, batch_size=train_params['batch_size'],
                             collate_fn=graph_collate_func_DGTNet_normalization_require, shuffle=False,
                             drop_last=True, num_workers=num_workers, pin_memory=False)

    # 测试模型
    model_interface.DGTNet_model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in test_loader:
            graph_name_list, adjacency_matrix, node_features, imf_1_matrices, imf_2_matrices, imf_3_matrices, imf_4_matrices, imf_5_matrices, y_true_normalization, _, _ = batch
            adjacency_matrix = adjacency_matrix.to(train_params['device'])
            node_features = node_features.to(train_params['device'])
            imf_1_matrices = imf_1_matrices.to(train_params['device'])
            imf_2_matrices = imf_2_matrices.to(train_params['device'])
            imf_3_matrices = imf_3_matrices.to(train_params['device'])
            imf_4_matrices = imf_4_matrices.to(train_params['device'])
            imf_5_matrices = imf_5_matrices.to(train_params['device'])
            y_true_normalization = y_true_normalization.to(train_params['device'])
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0

            y_pred_normalization = model_interface.DGTNet_model(
                node_features, batch_mask, adjacency_matrix, imf_1_matrices, imf_2_matrices, imf_3_matrices,
                imf_4_matrices, imf_5_matrices
            )
            predictions.append(y_pred_normalization.cpu().numpy())
            targets.append(y_true_normalization.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    print('y_pred_normalization.shape', y_pred_normalization.shape)
    print('y_true_normalization.shape', y_true_normalization.shape)
    print('predictions.shape',predictions.shape)
    print('targets.shape',targets.shape)

    # 计算RMSE
    rmse = calculate_rmse(targets, predictions)
    print("RMSE:", rmse)
    mae = calculate_mae(targets, predictions)
    print("MAE:", mae)
    mape = calculate_mape(targets, predictions)
    print("MAPE:", mape)
torch.cuda.empty_cache()

