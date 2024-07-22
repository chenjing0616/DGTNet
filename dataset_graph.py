import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.utils.rnn as rnn_utils

class DGTNet_Graph:
    """
    construct the DGTNet graph element (without normalized) as a class
    """
    # added for fitting the context in DGTNet
    def __init__(self, graph, label):
        assert isinstance(graph, dict)
        self.graph_name = graph['date']
        self.label = label
        self.node_features = graph['node_feat']
        self.imf_matrices_dict = graph['imf_matries_dict']
        self.adjacency_matrix = graph['adj_mat']


class DGTNet_normalized_Graph:
    """
    construct the DGTNet graph element (with normalized) as a class
    """
    # added for fitting the context in DGTNet with normalized data structure
    def __init__(self, graph, label):
        assert isinstance(graph, dict)
        self.graph_name = graph['date']
        self.label = label
        self.node_features = graph['node_feat']
        self.imf_matrices_dict = graph['imf_matries_dict']   # 5个228*228的邻接矩阵，所以论文里的公式是求和
        self.adjacency_matrix = graph['adj_mat']
        self.local_max_vec, self.local_min_vec = graph['node_local_MAX'], graph['node_local_MIN']
        self.global_max_vec, self.global_min_vec = graph['node_global_MAX'], graph['node_global_MIN']


class DGTNet_GraphDataSet(Dataset):
    def __init__(self, data_list):
        self.data_list = np.array(data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        if type(key) == slice:
            return DGTNet_GraphDataSet(self.data_list[key])
        return self.data_list[key]


def pad_array(array, shape):
    # padding for unfill entriess
    padded_array = np.zeros(shape, dtype=np.float)
    if len(shape) == 2:
        padded_array[:array.shape[0], :array.shape[1]] = array
    elif len(shape) == 3:
        padded_array[:array.shape[0], :array.shape[1], :] = array
    return padded_array


def construct_DGTNet_dataset(graph_list, label_list, normalization_require=False):
    """
    construct a TAST dataset with apply the class property to each elements.
    """
    assert isinstance(graph_list, list)
    assert isinstance(label_list, list)
    output = []
    if normalization_require:   # normalization
        print('Constructing an normalizated DGTNet dataset...')
        for (graph, label) in tqdm(zip(graph_list, label_list), total=len(graph_list)):
            output.append(DGTNet_normalized_Graph(graph, label))
    else:                       # no normalization
        print('Constructing an unnormalizated DGTNet dataset...')
        for (graph, label) in tqdm(zip(graph_list, label_list), total=len(graph_list)):
            output.append(DGTNet_Graph(graph, label))
    return DGTNet_GraphDataSet(output)


def graph_collate_func_DGTNet(batch):
    """
    A transpmation function that convert a batch (without normalizeds) into list of lists

    :params: batch
    """
    graph_name_list, adjacent_list, node_feature_list, label_list = [], [], [], [],
    # imf matrics
    imf_1_matrix_list = []
    imf_2_matrix_list = []
    imf_3_matrix_list = []
    imf_4_matrix_list = []
    imf_5_matrix_list = []

    for graph in batch:
        graph_name_list.append(graph.graph_name)
        adjacent_list.append(graph.adjacency_matrix)
        node_feature_list.append(graph.node_features)
        imf_1_matrix_list.append(graph.imf_matrices_dict['imf_1_matix'])
        imf_2_matrix_list.append(graph.imf_matrices_dict['imf_2_matix'])
        imf_3_matrix_list.append(graph.imf_matrices_dict['imf_3_matix'])
        imf_4_matrix_list.append(graph.imf_matrices_dict['imf_4_matix'])
        imf_5_matrix_list.append(graph.imf_matrices_dict['imf_5_matix'])

        if isinstance(graph.label, list):       # task number != 1
            label_list.append(graph.label)
        else:                                      # task number == 1
            label_list.append([graph.label])
    output_list = []
    output_list.append(graph_name_list)
    for graph_features in [adjacent_list, node_feature_list, imf_1_matrix_list, imf_2_matrix_list,
                          imf_3_matrix_list, imf_4_matrix_list, imf_5_matrix_list, label_list]:
        output_list.append(torch.from_numpy(np.array(graph_features)).float())
    return output_list


def graph_collate_func_DGTNet_normalization_require(batch):
    """
    A transpmation function that convert a batch (without normalizeds) into list of lists

    :params: batch
    """
    graph_name_list, adjacent_list, node_feature_list, label_list, local_max_vec, local_min_vec = [], [], [], [], [] ,[]
    # imf matrics
    imf_1_matrix_list = []
    imf_2_matrix_list = []
    imf_3_matrix_list = []
    imf_4_matrix_list = []
    imf_5_matrix_list = []

    for normalized_graph in batch:
        graph_name_list.append(normalized_graph.graph_name)
        adjacent_list.append(normalized_graph.adjacency_matrix)
        node_feature_list.append(normalized_graph.node_features)
        imf_1_matrix_list.append(normalized_graph.imf_matrices_dict['imf_1_matix'])
        imf_2_matrix_list.append(normalized_graph.imf_matrices_dict['imf_2_matix'])
        imf_3_matrix_list.append(normalized_graph.imf_matrices_dict['imf_3_matix'])
        imf_4_matrix_list.append(normalized_graph.imf_matrices_dict['imf_4_matix'])
        imf_5_matrix_list.append(normalized_graph.imf_matrices_dict['imf_5_matix'])
        local_max_vec.append(normalized_graph.local_max_vec)
        local_min_vec.append(normalized_graph.local_min_vec)

        if isinstance(normalized_graph.label, list):       # task number != 1
            label_list.append(normalized_graph.label)
        else:                                      # task number == 1
            label_list.append([normalized_graph.label])
    output_list = []
    output_list.append(graph_name_list)

    # Convert to PyTorch tensors
    adjacent_list = rnn_utils.pad_sequence([torch.from_numpy(adj) for adj in adjacent_list], batch_first=True)
    node_feature_list = rnn_utils.pad_sequence([torch.from_numpy(features) for features in node_feature_list],batch_first=True)
    imf_1_matrix_list = rnn_utils.pad_sequence([torch.from_numpy(imf) for imf in imf_1_matrix_list], batch_first=True)
    imf_2_matrix_list = rnn_utils.pad_sequence([torch.from_numpy(imf) for imf in imf_2_matrix_list], batch_first=True)
    imf_3_matrix_list = rnn_utils.pad_sequence([torch.from_numpy(imf) for imf in imf_3_matrix_list], batch_first=True)
    imf_4_matrix_list = rnn_utils.pad_sequence([torch.from_numpy(imf) for imf in imf_4_matrix_list], batch_first=True)
    imf_5_matrix_list = rnn_utils.pad_sequence([torch.from_numpy(imf) for imf in imf_5_matrix_list], batch_first=True)
    label_list = rnn_utils.pad_sequence([torch.tensor(label) for label in label_list], batch_first=True)

    for graph_features in [adjacent_list, node_feature_list, imf_1_matrix_list, imf_2_matrix_list,
                           imf_3_matrix_list, imf_4_matrix_list, imf_5_matrix_list, label_list,
                           local_max_vec, local_min_vec]:
        output_list.append(torch.from_numpy(np.array(graph_features)).float())
    return output_list
