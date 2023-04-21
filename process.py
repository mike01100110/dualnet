import os
import re
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch as th
# from sklearn.model_selection import ShuffleSplit
from utils import sys_normalized_adjacency,sparse_mx_to_torch_sparse_tensor,sys_normalized_adjacency_i
import pickle as pkl
import sys
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

#adapted from geom-gcn
# 定义一个名为parse_index_file的函数，接收一个文件名为参数。
def parse_index_file(filename):
    """Parse index file."""
    # 空列表index
    index = []
    # 打开文件并逐行读取，使用for循环遍历每一行。
    for line in open(filename):
        # 将读取的每一行的首尾空格去掉，将其转换为整数类型，然后将其添加到index列表中
        index.append(int(line.strip()))
    #     返回index列表，表示整个文件已经被处理完毕，其中包含所有转换后的整数。
    return index

# 定义一个名为sample_mask的函数，接受两个参数：idx表示需要设置为True的元素的索引，l表示数组的长度。
def sample_mask(idx, l):
    """Create mask."""
    # 创建一个长度为l的NumPy数组mask，其中所有元素初始化为0。
    mask = np.zeros(l)
    # 将索引为idx的元素设置为1，其他元素保持为0。
    mask[idx] = 1
    # 将mask数组转换为布尔型数组，并将其返回
    return np.array(mask, dtype=bool)

# 定义一个名为full_load_citation的函数，接受一个字符串类型的参数dataset_str，表示需要加载的数据集名称。
def full_load_citation(dataset_str):
    # 定义一个包含字符串的列表names，其中每个字符串代表数据集中的一个文件，分别为特征、标签、测试集特征、测试集标签、所有节点的特征、所有节点的标签和图结构。
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    # 在一个循环中，依次从文件中加载每个数据集文件，并将结果添加到objects列表中。如果当前的Python版本大于3.0，就用latin1编码方式打开文件，否则默认编码方式。
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    # 将objects列表中的数据按顺序依次赋值给7个变量，分别为x、y、tx、ty、allx、ally和graph。
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    # 调用parse_index_file函数，从测试索引文件中读取测试数据的索引，并将结果存储在test_idx_reorder变量中
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    # 将test_idx_reorder数组进行排序，结果存储在test_idx_range变量中
    test_idx_range = np.sort(test_idx_reorder)
    # 对于Citeseer数据集，由于存在孤立节点，需要进行修复。找到这些孤立节点，将其添加为全零向量到正确的位置
    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended
    # 将所有节点的特征和测试集的特征合并为一个特征矩阵，存储在features变量中。将测试集中的特征按照test_idx_range重新排序
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    #该行代码接受一个表示图的字典，并创建对应的邻接矩阵。nx 是 NetworkX 包的缩写，这是一个用于处理图的 Python 库
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # 该行代码垂直堆叠了两个 NumPy 数组 ally 和 ty，创建了一个新的数组 labels
    labels = np.vstack((ally, ty))
    # 该行代码通过根据 test_idx_reorder 索引重新排列 labels 数组的行来更新 labels 数组
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    # 该行代码从 NumPy 数组 test_idx_range 创建了一个新列表 idx_test
    idx_test = test_idx_range.tolist()
    # 该行代码使用 NumPy 数组 y 的长度创建了一个新的 range 对象 idx_train
    idx_train = range(len(y))
    # 该行代码从 NumPy 数组 y 的长度开始，创建了一个新的 range 对象，其范围为 len(y) 到 len(y)+500
    idx_val = range(len(y), len(y)+500)
    # 该行代码使用函数 sample_mask 和 idx_train range 对象创建了一个布尔掩码 train_mask
    train_mask = sample_mask(idx_train, labels.shape[0])
    # 该行代码使用函数 sample_mask 和 idx_val range 对象创建了一个布尔掩码 val_mask
    val_mask = sample_mask(idx_val, labels.shape[0])
    # 该行代码使用函数 sample_mask 和 idx_test 列表创建了一个布尔掩码 test_mask
    test_mask = sample_mask(idx_test, labels.shape[0])
    # 该行代码创建了一个与 labels 形状相同且值为零的 NumPy 数组 y_train
    y_train = np.zeros(labels.shape)
    # 该行代码创建了一个与 labels 形状相同且值为零的 NumPy 数组 y_val
    y_val = np.zeros(labels.shape)
    # 该行代码创建了一个与 labels 形状相同且值为零的 NumPy 数组 y_test
    y_test = np.zeros(labels.shape)
    # 该行代码通过将 train_mask 索引的行设置为 labels 对应的行来更新 y_train 数组
    y_train[train_mask, :] = labels[train_mask, :]
    # 该行代码通过将 val_mask 索引的行设置为 labels 对应的行来更新 y_val 数组
    y_val[val_mask, :] = labels[val_mask, :]
    # 该行代码通过将 test_mask 索引的行设置为 labels 对应的行来更新 y_test 数组
    y_test[test_mask, :] = labels[test_mask, :]
    # 行代码从函数中返回多个变量，包括邻接矩阵 adj、特征矩阵 features、标签矩阵 labels、训练集掩码 train_mask、验证集掩码 val_mask 和测试集掩码 test_mask
    # 这些变量都将用于执行图分类任务，其中邻接矩阵表示图中节点之间的连接关系，特征矩阵表示每个节点的特征，标签矩阵表示每个节点的真实类别，而掩码则用于标识数据集中哪些节点用于训练、验证和测试
    return adj, features, labels, train_mask, val_mask, test_mask


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    # 计算特征矩阵 features 中每个节点的度数，即每个节点与多少个其他节点相连
    rowsum = np.array(features.sum(1))
    # 处理节点度数为0的情况，将节点的度数为0的行转化为度数为1的行，以避免在接下来的行归一化操作中出现除以0的情况
    rowsum = (rowsum==0)*1+rowsum
    # 计算每个节点的度数的倒数，并将结果展开成一维数组
    r_inv = np.power(rowsum, -1).flatten()
    # 处理节点度数为0的情况，将度数为0的节点的度数倒数设置为0，以避免在接下来的矩阵相乘操作中出现inf值
    r_inv[np.isinf(r_inv)] = 0.
    # 创建一个对角矩阵，其中对角线上的元素为每个节点的度数的倒数
    r_mat_inv = sp.diags(r_inv)
    # 将特征矩阵进行行归一化操作，即将每个节点的特征向量除以该节点的度数，以确保每个节点的特征向量总和为1
    features = r_mat_inv.dot(features)
    # 返回处理后的特征矩阵
    return features


# 这个函数主要是用于加载图数据和相应的拆分数据，dataset_name：字符串，指定要加载的数据集的名称
# splits_file_path：字符串，指定拆分数据的文件路径，默认值为None，表示不加载拆分数据
def full_load_data(dataset_name, splits_file_path=None):
    # 判断要加载的数据集是引用数据集（即'cora'、'citeseer'或'pubmed'）还是自定义数据集
    if dataset_name in {'cora', 'citeseer', 'pubmed'}:
        # 如果是引用数据集，调用full_load_citation函数来加载数据。在加载数据时，将标签从one-hot形式转换为单个整数，并将节点特征从稀疏矩阵表示转换为密集矩阵表示。同时，创建一个有向图对象并返回
        adj, features, labels, _, _, _ = full_load_citation(dataset_name)
        labels = np.argmax(labels, axis=-1)
        features = features.todense()
        G = nx.DiGraph(adj)
    #  如果是自定义数据集，首先从文件中加载节点特征和标签，然后从另一个文件中加载邻接列表，使用这些信息构建一个有向图对象。然后对该图进行预处理，将邻接矩阵进行归一化处理，并将节点特征和标签转换为PyTorch张量。最后，从拆分文件中加载训练、验证和测试掩码，并返回所需的数据
    else:
        # 第一行代码定义了一个变量graph_adjacency_list_file_path，用于存储边列表文件的路径。这个文件的格式是每行两个数字，表示一条边连接的两个节点的编号
        graph_adjacency_list_file_path = os.path.join('new_data', dataset_name, 'out1_graph_edges.txt')
        # 用于存储节点特征和标签文件的路径。这个文件的格式是每行多个数字，第一个数字表示节点编号，后面的数字表示节点的特征和标签
        graph_node_features_and_labels_file_path = os.path.join('new_data', dataset_name,
                                                                'out1_node_feature_label.txt')
        # 创建了一个空的有向图（DiGraph）对象G
        G = nx.DiGraph()
        # 创建了两个空的字典对象graph_node_features_dict和graph_labels_dict，用于存储节点特征和标签
        graph_node_features_dict = {}
        graph_labels_dict = {}

        if dataset_name=='film':
             with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])

        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        features = np.array(
            [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
        labels = np.array(
            [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
    features = preprocess_features(features)
    g = adj
  
    with np.load(splits_file_path) as splits_file:
        train_mask = splits_file['train_mask']
        val_mask = splits_file['val_mask']
        test_mask = splits_file['test_mask']
    
    num_features = features.shape[1]
    num_labels = len(np.unique(labels))
    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

    features = th.FloatTensor(features)
    labels = th.LongTensor(labels)
    train_mask = th.BoolTensor(train_mask)
    val_mask = th.BoolTensor(val_mask)
    test_mask = th.BoolTensor(test_mask)

    adj = sys_normalized_adjacency(g)
    adj_i = sys_normalized_adjacency_i(g)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj_i = sparse_mx_to_torch_sparse_tensor(adj_i)

    # adj：PyTorch稀疏张量，表示图的邻接矩阵
    # adj_i：PyTorch稀疏张量，表示图的邻接矩阵的逆矩阵
    # features：PyTorch张量，表示节点特征矩阵
    # labels：PyTorch张量，表示节点标签
    # train_mask：PyTorch布尔张量，表示用于训练的节点的掩码
    # val_mask：PyTorch布尔张量，表示用于验证的节点的掩码
    # test_mask：PyTorch布尔张量，表示用于测试的节点的掩码
    # num_features：整数，表示节点特征的维数
    # num_labels：整数，表示节点标签的种类数
    return adj,adj_i, features, labels, train_mask, val_mask, test_mask, num_features, num_labels
