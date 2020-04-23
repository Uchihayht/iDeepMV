# coding=utf-8 
import sys
import os
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from urllib import urlretrieve
# import cPickle as pickle
# import gzip
# import theano
# import lasagne
# from lasagne import layers
# from lasagne.updates import nesterov_momentum
# from nolearn.lasagne import NeuralNet
# from nolearn.lasagne import visualize

import numpy as np
import tensorflow as tf
import random
import pdb
import datetime
import xlrd
import xlwt
import sendmessage
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from keras.models import Sequential
import keras.layers.core as core
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam
from keras.layers import normalization, Lambda, GlobalMaxPooling2D
from keras.layers import LSTM, Bidirectional, Reshape, Layer
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import merge, Input, TimeDistributed
from keras.regularizers import l1, l2
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.constraints import maxnorm

from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.utils import class_weight
from keras import objectives
from keras import backend as K

# import utils

length = 2700

sampleNum = 0

codon_table = {
    'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A', 'CGU': 'R', 'CGC': 'R',
    'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R', 'UCU': 'S', 'UCC': 'S',
    'UCA': 'S', 'UCG': 'S', 'AGU': 'S', 'AGC': 'S', 'AUU': 'I', 'AUC': 'I',
    'AUA': 'I', 'UUA': 'L', 'UUG': 'L', 'CUU': 'L', 'CUC': 'L', 'CUA': 'L',
    'CUG': 'L', 'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G', 'GUU': 'V',
    'GUC': 'V', 'GUA': 'V', 'GUG': 'V', 'ACU': 'T', 'ACC': 'T', 'ACA': 'T',
    'ACG': 'T', 'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P', 'AAU': 'N',
    'AAC': 'N', 'GAU': 'D', 'GAC': 'D', 'UGU': 'C', 'UGC': 'C', 'CAA': 'Q',
    'CAG': 'Q', 'GAA': 'E', 'GAG': 'E', 'CAU': 'H', 'CAC': 'H', 'AAA': 'K',
    'AAG': 'K', 'UUU': 'F', 'UUC': 'F', 'UAU': 'Y', 'UAC': 'Y', 'AUG': 'M',
    'UGG': 'W'
}


def classifaction_report_csv(report, name):
    report_data = []
    lines = report.split('\n')
    for line in lines[0:4]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    writer = pd.ExcelWriter(name)
    dataframe.to_excel(writer, 'page_1', float_format='%.5f')
    writer.save()


def cinResultByExcel(name):
    workbook = xlrd.open_workbook(name)
    sheet = workbook.sheet_by_index(0)
    columns = []
    for i in range(sheet.nrows):
        columns.append(int(sheet.cell(i, 0).value))
    return columns


def cinResultByExcel1(name):
    workbook = xlrd.open_workbook(name)
    sheet = workbook.sheet_by_index(0)
    columns = []
    for i in range(sheet.nrows):
        columns.append(sheet.row_values(i))
    return columns


def coutResultByExcel(preds, name):
    preds = pd.DataFrame(preds)
    writer = pd.ExcelWriter(name)
    preds.to_excel(writer, 'page_1', float_format='%.5f')  # float_format 控制精度
    writer.save()


# 将RNA序列转化为3条氨基酸序列，
def translateTheSeq(seq):
    seq1 = ''
    seq2 = ''
    seq3 = ''
    for frame in range(3):
        prot = ''
        for i in range(frame, len(seq), 3):
            codon = seq[i:i + 3]
            if codon in codon_table:
                prot = prot + codon_table[codon]
            else:
                prot = prot + 'O'
        if frame == 0:
            seq1 = prot
        elif frame == 1:
            seq2 = prot
        elif frame == 2:
            seq3 = prot
    seq1 = seq1 + seq2 + seq3
    return seq1


# 将氨基酸序列转化为二肽成分一位数组
def CountTwoPeptide(seq):
    alpha = 'ACDEFGHIKLMNPQRSTVWYO'
    number = np.ones((1, 440)) * 0
    for i in range(len(seq) - 1):
        if alpha.index(seq[i]) == 20 and alpha.index(seq[i + 1]) == 20:
            continue
        else:
            number[0][alpha.index(seq[i]) * 21 + alpha.index(seq[i + 1])] += 1
    # print(number)
    return number


# 将上面的一维数组转化为二肽直方图
def translate(seq):
    number = np.ones([440, 30]) * 0
    for i in range(len(seq[0])):
        tag = seq[0][i]
        if tag == 0:
            continue
        else:
            number[i, 0:int(tag)] = 1
    return number


# 读取文件的RNA序列，转化为以RNA名称为索引，RNA序列为主体的一个二维表格结构
def read_fasta_file(fasta_file):
    seq_dict = {}
    fp = open(fasta_file, 'r')
    name = ''
    for line in fp:
        line = line.rstrip()  # 删除line尾部的空格
        # distinguish header from sequence
        if line[0] == '>':  # or line.startswith('>')
            # it is the header
            name = line[1:]  # discarding the initial >
            seq_dict[name] = ''
        else:
            seq_dict[name] = seq_dict[name] + line.upper()  # 将字符串中的小写字母转化为大写字母
    fp.close()

    return seq_dict


# 读取文件的RNA序列，转化为以RNA名称为索引，RNA序列为主体的一个二维表格结构
def read_fasta_file_new(fasta_file='../data/UTR_hg19.fasta'):
    seq_dict = {}
    fp = open(fasta_file, 'r')
    name = ''
    for line in fp:
        line = line.rstrip()
        # distinguish header from sequence
        if line[0] == '>':  # or line.startswith('>')
            # it is the header
            name = line[1:].split()[0]  # discarding the initial >
            seq_dict[name] = ''
        else:
            seq_dict[name] = seq_dict[name] + line.upper()
    fp.close()

    return seq_dict


# 从结合文件中读取信息，建立三个字典，为（1）以RNA为索引的字典，（2）rna序列字典，其中名称为索引，（3）蛋白质集合
def load_rnacomend_data(datadir='../data/'):
    pair_file = datadir + 'interactions_HT.txt'
    # rbp_seq_file = datadir + 'rbps_HT.fa'
    rna_seq_file = datadir + 'utrs.fa'

    rna_seq_dict = read_fasta_file(rna_seq_file)
    protein_set = set()
    inter_pair = {}
    new_pair = {}
    with open(pair_file, 'r') as fp:
        for line in fp:
            values = line.rstrip().split()
            protein = values[0]
            protein_set.add(protein)
            rna = values[1]
            if sampleNum != 0:
                if len(new_pair.setdefault(protein, [])) < sampleNum:
                    # 以结合蛋白为索引，向字典里加入RNA序列，所以这个字典一共只有68个数据集合，代表68个结合蛋白
                    new_pair.setdefault(protein, []).append(rna)
                    # 以RNA为索引，建立字典，内容为每种RNA可结合的蛋白序列群
                    inter_pair.setdefault(rna, []).append(protein)
            else:
                inter_pair.setdefault(rna, []).append(protein)

    # 返回的是取决于样本数量的（1）以RNA为索引的字典，（2）rna序列字典，其中名称为索引，（3）蛋白质集合
    return inter_pair, rna_seq_dict, protein_set


# 将load_rnacomend_data中的RNA索引字典中的RNA和无标签RNA整合起来，形成一个大集合。统计这个集合中所有RNA各自的标签
def get_rnarecommend(inter_pair_dict, rna_seq_dict, protein_list):
    data = {}
    labels = []  # 里面装的是所有RNA的标签二维数组
    rna_seqs = []  # 里面装的是所有RNA序列

    protein_list.append("negative")  # 向蛋白质集合中加入空标签
    f = open("protein_list.txt", 'w')
    for prote in protein_list:
        f.write(prote + "\n")
    f.close()
    print(protein_list)
    all_hg19_utrs = read_fasta_file_new()
    remained_rnas = list(set(all_hg19_utrs.keys()) - set(inter_pair_dict.keys()))  # 这个remained_rnas是无标签rna集合
    # pdb.set_trace()
    for rna, protein in inter_pair_dict.items():
        rna_seq = rna_seq_dict[rna]
        rna_seq = rna_seq.replace('T', 'U')
        init_labels = np.array([0] * len(protein_list))  # 初始化此条RNA序列的标签信息，即创建一个68维的值为0的向量
        inds = []
        for pro in protein:
            inds.append(protein_list.index(pro))
        init_labels[np.array(inds)] = 1
        labels.append(init_labels)
        rna_seqs.append(rna_seq)

    # 这里开始处理无标签RNA，将它们加入rna_seqs,再将它们的无标签信息加入到labels中
    max_num_targets = np.sum(labels, axis=0).max()
    print(max_num_targets)
    random.shuffle(remained_rnas)
    for rna in remained_rnas[:max_num_targets]:
        rna_seq = all_hg19_utrs[rna]
        rna_seq = rna_seq.replace('T', 'U')
        rna_seqs.append(rna_seq)
        init_labels = np.array([0] * (len(protein_list) - 1) + [1])
        labels.append(init_labels)

    data["seq"] = rna_seqs
    data["Y"] = np.array(labels)

    return data


# 将传进来的RNA序列按照指定大小进行裁剪或者拉伸，如果位数不够，则用B字母进行填充
def padding_sequence(seq, max_len=length, repkey='B'):
    seq_len = len(seq)
    if seq_len < max_len:
        gap_len = max_len - seq_len
        new_seq = seq + repkey * gap_len
    else:
        new_seq = seq[:max_len]
    return new_seq


# 将RNA序列先转化为氨基酸序列，再进行one-hot编码，返回的是此条RNA序列的3个氨基酸one-hot编码二维数组
def get_RNA_seq_concolutional_array(seq, motif_len=10):
    # print(seq)
    seq1 = translateTheSeq(seq)
    alpha = 'ACDEFGHIKLMNPQRSTVWY'

    half_len = motif_len / 2
    half_len = int(half_len)

    row = (len(seq1) + 2 * half_len)
    new_array = np.zeros((row, 20))

    for i in range(half_len):
        new_array[i] = np.array([0.05] * 20)

    for i in range(row - half_len, row):
        new_array[i] = np.array([0.05] * 20)

    # pdb.set_trace()
    for i, val in enumerate(seq1):
        i = i + half_len
        if val not in 'ACDEFGHIKLMNPQRSTVWY':
            new_array[i] = np.array([0.05] * 20)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
    return new_array


def get_RNA_seq_concolutional_array_Component(seq, motif_len=10):
    # print(seq)
    seq11 = translateTheSeq(seq)
    seq1 = translate(CountTwoPeptide(seq11))
    return seq1


def get_RNA_seq_concolutional_array_Pan(seq, motif_len=10):
    seq = seq.replace('U', 'T')
    alpha = 'ACGT'
    half_len = motif_len / 2
    half_len = int(half_len)
    row = (len(seq) + 2 * half_len)
    new_array = np.zeros((row, 4))
    for i in range(half_len):
        new_array[i] = np.array([0.25] * 4)

    for i in range(row - half_len, row):
        new_array[i] = np.array([0.25] * 4)

    # pdb.set_trace()
    for i, val in enumerate(seq):
        i = i + half_len
        if val not in 'ACGT':
            new_array[i] = np.array([0.25] * 4)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
            # data[key] = new_array
    return new_array


# 将所有RNA序列转化为三条氨基酸one-hot编码数组，组装成包返回
def get_bag_data_1_channel(seqs, labels, max_len=length):
    bags = []
    for seq in seqs:
        bag_seq = padding_sequence(seq, max_len=max_len)
        tri_fea = get_RNA_seq_concolutional_array(bag_seq)
        bags.append(np.array(tri_fea))
    return np.array(bags), np.array(labels)


def get_bag_data_1_channel_Component(seqs, labels, max_len=length):
    bags = []
    for seq in seqs:
        bag_seq = padding_sequence(seq, max_len=max_len)
        seq1 = get_RNA_seq_concolutional_array_Component(bag_seq)
        bags.append(np.array(seq1))
    return np.array(bags), np.array(labels)


def get_bag_data_1_channel_Pan(seqs, labels, max_len=length):
    bags = []
    for seq in seqs:
        bag_seq = padding_sequence(seq, max_len=max_len)
        tri_fea = get_RNA_seq_concolutional_array_Pan(bag_seq)
        bags.append(np.array(tri_fea))
    return np.array(bags), np.array(labels)


# 以分割方法得到的序列，来将整个数据集分割为训练集和测试集，打包各自的氨基酸序列和标签，返回
def get_all_rna_mildata(seqs, labels, training_val_indice, train_val_label, test_indice, test_label, max_len=length):
    train_seqs = []
    for val in training_val_indice:
        train_seqs.append(seqs[val])
    train_bags, train_labels = get_bag_data_1_channel(train_seqs, train_val_label, max_len=max_len)

    test_seqs = []
    for val in test_indice:
        test_seqs.append(seqs[val])
    test_bags, test_labels = get_bag_data_1_channel(test_seqs, test_label, max_len=max_len)

    return train_bags, train_labels, test_bags, test_labels


def get_all_rna_mildata_Component(seqs, labels, training_val_indice, train_val_label, test_indice, test_label,
                                  max_len=length):
    train_seqs = []
    for val in training_val_indice:
        train_seqs.append(seqs[val])
    train_bags, train_labels = get_bag_data_1_channel_Component(train_seqs, train_val_label, max_len=max_len)

    test_seqs = []
    for val in test_indice:
        test_seqs.append(seqs[val])
    test_bags, test_labels = get_bag_data_1_channel_Component(test_seqs, test_label, max_len=max_len)

    return train_bags, train_labels, test_bags, test_labels


def get_all_rna_mildata_Pan(seqs, labels, training_val_indice, train_val_label, test_indice, test_label,
                            max_len=length):
    train_seqs = []
    for val in training_val_indice:
        train_seqs.append(seqs[val])
    train_bags_Pan, train_labels_Pan = get_bag_data_1_channel_Pan(train_seqs, train_val_label, max_len=max_len)

    test_seqs = []
    for val in test_indice:
        test_seqs.append(seqs[val])
    test_bags_Pan, test_labels_Pan = get_bag_data_1_channel_Pan(test_seqs, test_label, max_len=max_len)

    return train_bags_Pan, train_labels_Pan, test_bags_Pan, test_labels_Pan


# 为了防止内存不够，采用多视角数据分批录入方法
def getYData(seqs, labels, training_val_indice, test_indice, train_val_label, test_label):
    train_bags, train_labels, test_bags, test_labels = get_all_rna_mildata(seqs, labels, training_val_indice,
                                                                           train_val_label, test_indice, test_label)
    return train_bags, train_labels, test_bags, test_labels


def getYData_Component(seqs, labels, training_val_indice, test_indice, train_val_label, test_label):
    train_bags, train_labels, test_bags, test_labels = get_all_rna_mildata_Component(seqs, labels, training_val_indice,
                                                                                     train_val_label, test_indice,
                                                                                     test_label)
    return train_bags, train_labels, test_bags, test_labels


def getPData(seqs, labels, training_val_indice, test_indice, train_val_label, test_label):
    train_bags_Pan, train_labels_Pan, test_bags_Pan, test_labels_Pan = get_all_rna_mildata_Pan(seqs, labels,
                                                                                               training_val_indice,
                                                                                               train_val_label,
                                                                                               test_indice, test_label)
    return train_bags_Pan, train_labels_Pan, test_bags_Pan, test_labels_Pan


# 主要为获取分割训练集和测试集的序号包，然后分配给getYDate，从数据库中获取真正的数据
def get_all_data():
    inter_pair_dict, rna_seq_dict, protein_set = load_rnacomend_data()
    protein_list = []
    for protein in protein_set:
        protein_list.append(protein)
    data = get_rnarecommend(inter_pair_dict, rna_seq_dict, protein_list)
    labels = data["Y"]
    seqs = data["seq"]
    x_index = range(len(labels))
    training_val_indice, test_indice, train_val_label, test_label = train_test_split(x_index, labels, train_size=0.8,
                                                                                     stratify=labels)
    return seqs, labels, training_val_indice, test_indice, train_val_label, test_label, protein_list


def set_cnn_model(input_dim=20, input_length=2710, nbfilter=101):
    model = Sequential()
    # model.add(brnn)

    model.add(Conv1D(input_dim=input_dim, input_length=input_length,
                     nb_filter=nbfilter,
                     filter_length=10,
                     border_mode="valid",
                     # activation="relu",
                     subsample_length=1))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_length=3))

    model.add(Flatten())

    model.add(Dropout(0.5))
    model.add(Dense(nbfilter * 2, activation='relu'))
    model.add(BatchNormalization(name='feature'))
    model.add(Dropout(0.5))
    return model


def set_cnn_model_Component(input_dim=30, input_length=440, nbfilter=101):
    model = Sequential()
    # model.add(brnn)

    model.add(Conv1D(input_dim=input_dim, input_length=input_length,
                     nb_filter=nbfilter,
                     filter_length=10,
                     border_mode="valid",
                     # activation="relu",
                     subsample_length=1))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # model.add(MaxPooling1D(pool_length=3))

    model.add(Flatten())

    model.add(Dropout(0.5))
    model.add(Dense(nbfilter * 2, activation='relu'))
    model.add(BatchNormalization(name='feature'))
    model.add(Dropout(0.5))

    return model


def set_cnn_model_Pan(input_dim=4, input_length=2710, nbfilter=101):
    model = Sequential()
    # model.add(brnn)

    model.add(Conv1D(input_dim=input_dim, input_length=input_length,
                     nb_filter=nbfilter,
                     filter_length=10,
                     border_mode="valid",
                     # activation="relu",
                     subsample_length=1))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_length=3))

    model.add(Flatten())

    model.add(Dropout(0.5))
    model.add(Dense(nbfilter * 2, activation='relu'))
    model.add(BatchNormalization(name='feature'))
    model.add(Dropout(0.5))

    return model


def coutSameAndDifferend(preds, result):
    all = len(preds) * len(preds[1])
    total = 0
    same = 0
    error = 0
    forget = 0
    for i in range(len(preds)):
        for j in range(len(preds[i])):
            if result[i][j] == 1 and preds[i][j] == 1:
                total += 1
                same += 1
            elif result[i][j] == 1 and preds[i][j] == 0:
                total += 1
                forget += 1
            elif result[i][j] == 0 and preds[i][j] == 1:
                error += 1
    return total, same, error, all, forget


def GetLabelFeature(trainFeature, trainLabel, testFeature, testLabel, name):
    coutResultByExcel(trainFeature, "label/" + name + "trainFeature.xlsx");
    coutResultByExcel(trainLabel, "label/" + name + "trainLabel.xlsx");
    coutResultByExcel(testFeature, "label/" + name + "testFeature.xlsx");
    coutResultByExcel(testLabel, "label/" + name + "testLabel.xlsx");


def run_mlcnn():
    daytime = datetime.datetime.now()
    daytime = str(daytime)
    daytime = daytime.replace(':', '-')
    path = "../result/" + daytime + ".txt"

    seqs, labels, training_val_indice, test_indice, train_val_label, test_label, protein_list = get_all_data()

    # 二肽视角
    sendmessage.runmessage("starting Amino acid Component cin")
    train_bags_Yang, train_labels_Yang, test_bags_Yang, test_labels_Yang = getYData_Component(seqs, labels,
                                                                                              training_val_indice,
                                                                                              test_indice,
                                                                                              train_val_label,
                                                                                              test_label)

    cclf_Yang = set_cnn_model_Component(nbfilter=101)
    cclf_Yang.add(Dense(68, activation='sigmoid'))
    cclf_Yang.compile(optimizer=Adam(), loss='binary_crossentropy')  # 'mean_squared_error'

    sendmessage.runmessage("starting Amino acid Component train")
    cclf_Yang.fit(train_bags_Yang, train_labels_Yang, batch_size=64, nb_epoch=20, verbose=0, class_weight='auto')


    #保存模型
    cclf_Yang.save("mymodel/二肽.h5",True,True)


    print(test_bags_Yang.shape)
    preds_Yang = cclf_Yang.predict(test_bags_Yang)

    layer_name = 'feature'  # 获取层的名称
    intermediate_layer_model = Model(inputs=cclf_Yang.input, outputs=cclf_Yang.get_layer(layer_name).output)  # 创建的新模型
    intermediate_output1 = intermediate_layer_model.predict(train_bags_Yang)
    intermediate_output2 = intermediate_layer_model.predict(test_bags_Yang)

    # 提取标签特征
    #GetLabelFeature(intermediate_output1, train_labels_Yang, intermediate_output2, test_labels_Yang, "二肽");

    del train_bags_Yang, train_labels_Yang

    # 输出模块
    f = open(path, 'w')
    f.write("开始时间：" + daytime + "\n")
    f.write("样本数：" + str(sampleNum) + "\n")
    f.write("此版本将二肽和氨基酸的三个分视角取消，合并为一个大的视角，此版本加入了输出个RBP的识别精度\n")
    f.write("二肽视角" + "\n")
    f.write("结束时间：" + str(datetime.datetime.now()) + "\n")
    f.write("Macro-AUC:" + str(roc_auc_score(test_labels_Yang, preds_Yang, average='macro')) + "\n")
    f.write("Micro-AUC:" + str(roc_auc_score(test_labels_Yang, preds_Yang, average='micro')) + "\n")
    f.write("weight-AUC:" + str(roc_auc_score(test_labels_Yang, preds_Yang, average='weighted')) + "\n")

    preds_Yang[preds_Yang >= 0.2] = 1
    preds_Yang[preds_Yang < 0.2] = 0

    f11 = f1_score(test_labels_Yang, preds_Yang, average='macro')
    f12 = f1_score(test_labels_Yang, preds_Yang, average='micro')
    f13 = f1_score(test_labels_Yang, preds_Yang, average='weighted')

    mydata = np.ones((68, 3)) * 0
    for i in range(0, 68):
        TP = 0
        FP = 0
        FN = 0
        for j in range(0, len(test_labels_Yang[:, i])):
            if test_labels_Yang[j, i] == 1 and preds_Yang[j, i] == 1:
                TP += 1
            elif test_labels_Yang[j, i] == 0 and preds_Yang[j, i] == 1:
                FP += 1
            elif test_labels_Yang[j, i] == 1 and preds_Yang[j, i] == 0:
                FN += 1

        if (TP + FP) == 0:
            precision = 0;
        else:
            precision = TP / (TP + FP)
        if (TP + FN) == 0:
            recall = 0;
        else:
            recall = TP / (TP + FN);
        if (2 * TP + FP + FN) == 0:
            f1score = 0;
        else:
            f1score = 2 * TP / (2 * TP + FP + FN)

        mydata[i][0] = precision
        mydata[i][1] = recall
        mydata[i][2] = f1score
    coutResultByExcel(mydata, '二肽精确度.xlsx')

    f.write("\n")
    f.write("阈值：" + str(0.2) + "\n")
    f.write("f1-macro:" + str(f11) + "\n")
    f.write("f1-micro:" + str(f12) + "\n")
    f.write("f1-weighted:" + str(f13) + "\n")
    f.write("\n")
    f.close()

    # 氨基酸视角
    sendmessage.runmessage("starting Amino acid cin")
    train_bags, train_labels, test_bags, test_labels = getYData(seqs, labels, training_val_indice, test_indice,
                                                                train_val_label, test_label)
    cclf = set_cnn_model(nbfilter=101)
    cclf.add(Dense(68, activation='sigmoid'))

    cclf.compile(optimizer=Adam(), loss='binary_crossentropy')  # 'mean_squared_error'

    sendmessage.runmessage("starting Amino acid train")
    cclf.fit(train_bags, train_labels, batch_size=64, nb_epoch=20, verbose=0, class_weight='auto')

    cclf.save("mymodel/氨基酸.h5", True, True)
    preds = cclf.predict(test_bags)

    layer_name = 'feature'  # 获取层的名称
    intermediate_layer_model = Model(inputs=cclf.input, outputs=cclf.get_layer(layer_name).output)  # 创建的新模型
    intermediate_output1 = intermediate_layer_model.predict(train_bags)
    intermediate_output2 = intermediate_layer_model.predict(test_bags)

    # 提取标签特征
    #GetLabelFeature(intermediate_output1, train_labels, intermediate_output2, test_labels_Yang, "氨基酸");

    del train_bags, train_labels

    # 输出模块
    f = open(path, 'a+')
    f.write("氨基酸视角" + "\n")
    f.write("结束时间：" + str(datetime.datetime.now()) + "\n")
    f.write("Macro-AUC:" + str(roc_auc_score(test_labels, preds, average='macro')) + "\n")
    f.write("Micro-AUC:" + str(roc_auc_score(test_labels, preds, average='micro')) + "\n")
    f.write("weight-AUC:" + str(roc_auc_score(test_labels, preds, average='weighted')) + "\n")

    preds[preds >= 0.2] = 1
    preds[preds < 0.2] = 0

    f11 = f1_score(test_labels, preds, average='macro')
    f12 = f1_score(test_labels, preds, average='micro')
    f13 = f1_score(test_labels, preds, average='weighted')

    mydata = np.ones((68, 3)) * 0
    for i in range(0, 68):
        TP = 0
        FP = 0
        FN = 0
        for j in range(0, len(test_labels[:, i])):
            if test_labels[j, i] == 1 and preds[j, i] == 1:
                TP += 1
            elif test_labels[j, i] == 0 and preds[j, i] == 1:
                FP += 1
            elif test_labels[j, i] == 1 and preds[j, i] == 0:
                FN += 1
        if (TP + FP) == 0:
            precision = 0;
        else:
            precision = TP / (TP + FP)
        if (TP + FN) == 0:
            recall = 0;
        else:
            recall = TP / (TP + FN);
        if (2 * TP + FP + FN) == 0:
            f1score = 0;
        else:
            f1score = 2 * TP / (2 * TP + FP + FN)

        mydata[i][0] = precision
        mydata[i][1] = recall
        mydata[i][2] = f1score
    coutResultByExcel(mydata, '氨基酸精确度.xlsx')

    f.write("\n")
    f.write("阈值：" + str(0.2) + "\n")
    f.write("f1-macro:" + str(f11) + "\n")
    f.write("f1-micro:" + str(f12) + "\n")
    f.write("f1-weighted:" + str(f13) + "\n")
    f.write("\n")
    f.close()

    # RNA视角
    sendmessage.runmessage("starting RNA cin")
    train_bags_Pan, train_labels_Pan, test_bags_Pan, test_labels_Pan = getPData(seqs, labels, training_val_indice,
                                                                                test_indice, train_val_label,
                                                                                test_label)
    clf = set_cnn_model_Pan(nbfilter=101)
    clf.add(Dense(68, activation='sigmoid'))

    clf.compile(optimizer=Adam(), loss='binary_crossentropy')  # 'mean_squared_error'

    sendmessage.runmessage("starting RNA train")
    clf.fit(train_bags_Pan, train_labels_Pan, batch_size=64, nb_epoch=20, verbose=0, class_weight='auto')
    clf.save("mymodel/RNA.h5", True, True)
    preds_Pan = clf.predict(test_bags_Pan)

    layer_name = 'feature'  # 获取层的名称
    intermediate_layer_model = Model(inputs=clf.input, outputs=clf.get_layer(layer_name).output)  # 创建的新模型
    intermediate_output1 = intermediate_layer_model.predict(train_bags_Pan)
    intermediate_output2 = intermediate_layer_model.predict(test_bags_Pan)
    # 提取标签特征
    #GetLabelFeature(intermediate_output1, train_labels_Pan, intermediate_output2, test_labels_Yang, "RNA");

    del train_bags_Pan, train_labels_Pan

    f = open(path, 'a+')
    f.write("RNA视角" + "\n")
    f.write("结束时间：" + str(datetime.datetime.now()) + "\n")
    f.write("Macro-AUC:" + str(roc_auc_score(test_labels_Pan, preds_Pan, average='macro')) + "\n")
    f.write("Micro-AUC:" + str(roc_auc_score(test_labels_Pan, preds_Pan, average='micro')) + "\n")
    f.write("weight-AUC:" + str(roc_auc_score(test_labels_Pan, preds_Pan, average='weighted')) + "\n")
    preds_Pan[preds_Pan >= 0.2] = 1
    preds_Pan[preds_Pan < 0.2] = 0
    f11 = f1_score(test_labels_Pan, preds_Pan, average='macro')
    f12 = f1_score(test_labels_Pan, preds_Pan, average='micro')
    f13 = f1_score(test_labels_Pan, preds_Pan, average='weighted')
    mydata = np.ones((68, 3)) * 0
    for i in range(0, 68):
        TP = 0
        FP = 0
        FN = 0
        for j in range(0, len(test_labels_Pan[:, i])):
            if test_labels_Pan[j, i] == 1 and preds_Pan[j, i] == 1:
                TP += 1
            elif test_labels_Pan[j, i] == 0 and preds_Pan[j, i] == 1:
                FP += 1
            elif test_labels_Pan[j, i] == 1 and preds_Pan[j, i] == 0:
                FN += 1
        if (TP + FP) == 0:
            precision = 0;
        else:
            precision = TP / (TP + FP)
        if (TP + FN) == 0:
            recall = 0;
        else:
            recall = TP / (TP + FN);
        if (2 * TP + FP + FN) == 0:
            f1score = 0;
        else:
            f1score = 2 * TP / (2 * TP + FP + FN)
        mydata[i][0] = precision
        mydata[i][1] = recall
        mydata[i][2] = f1score
    coutResultByExcel(mydata, 'RNA精确度.xlsx')
    f.write("\n")
    f.write("阈值：" + str(0.2) + "\n")
    f.write("f1-macro:" + str(f11) + "\n")
    f.write("f1-micro:" + str(f12) + "\n")
    f.write("f1-weighted:" + str(f13) + "\n")
    f.write("\n")
    f.close()

    # 遍历寻找最优参数，记录最佳f1-score
    sendmessage.runmessage("starting search the best result")
    f = open(path, 'a+')
    linshi = np.ones((len(preds_Yang), 68)) * 0
    for i in range(len(preds_Yang)):
        for j in range(0, 68):
            vote = preds_Yang[i][j] + preds[i][j] + preds_Pan[i][j]
            if (vote > 1):
                linshi[i][j] = 1
            else:
                linshi[i][j] = 0
    x1 = f1_score(test_labels_Pan, linshi, average='macro')
    x2 = f1_score(test_labels_Pan, linshi, average='micro')
    x3 = f1_score(test_labels_Pan, linshi, average='weighted')
    mydata = np.ones((68, 3)) * 0
    for i in range(0, 68):
        TP = 0
        FP = 0
        FN = 0
        for j in range(0, len(test_labels_Pan[:, i])):
            if test_labels_Pan[j, i] == 1 and linshi[j, i] == 1:
                TP += 1
            elif test_labels_Pan[j, i] == 0 and linshi[j, i] == 1:
                FP += 1
            elif test_labels_Pan[j, i] == 1 and linshi[j, i] == 0:
                FN += 1

        if (TP + FP) == 0:
            precision = 0;
        else:
            precision = TP / (TP + FP)
        if (TP + FN) == 0:
            recall = 0;
        else:
            recall = TP / (TP + FN);
        if (2 * TP + FP + FN) == 0:
            f1score = 0;
        else:
            f1score = 2 * TP / (2 * TP + FP + FN)
        mydata[i][0] = precision
        mydata[i][1] = recall
        mydata[i][2] = f1score
    coutResultByExcel(mydata, '投票精确度.xlsx')
    f.write("投票结果：\n")
    f.write("最佳f1-macro为：" + str(x1) + "\n")
    f.write("最佳f1-micro为：" + str(x2) + "\n")
    f.write("最佳f1-weighted为：" + str(x3) + "\n")
    f.close()
    sendmessage.runmessage("finish")


run_mlcnn()
