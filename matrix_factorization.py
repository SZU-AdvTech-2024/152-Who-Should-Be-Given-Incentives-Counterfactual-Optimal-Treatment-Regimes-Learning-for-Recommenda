# -*- coding: utf-8 -*-
# 导入SciPy库中的稀疏矩阵模块，并将其命名为sps。该模块用于处理稀疏矩阵，提供有效的存储和操作功能。
import scipy.sparse as sps

# 导入NumPy库，并将其命名为np。NumPy是用于科学计算的基础库，提供支持多维数组和各种数学函数。
import numpy as np

# 导入PyTorch库，这是一个用于深度学习的流行框架，支持张量计算和自动微分。
import torch

# 设置随机种子为2020，以确保结果的可重复性。在深度学习中，随机种子用于初始化权重和数据拆分等过程。
torch.manual_seed(2020)

# 从PyTorch库中导入神经网络模块nn。这个模块提供了构建神经网络所需的各种组件（如层、损失函数等）。
from torch import nn

# 导入PyTorch的功能性API，命名为F。这个模块包含了一些常用的函数，比如激活函数和损失函数，可以用于构建模型时的计算。
import torch.nn.functional as F

# 导入Python调试器模块pdb。这个模块可以在代码中设置断点，方便进行调试和检查程序的运行状态。
import pdb

# 样本生成函数
def generate_total_sample(num_user, num_item):
    '''
    num_user: 用户数量
    num_item: 物品数量
    返回值: 一个(num_user*num_item,2)的二维数组
    '''
    sample = [] # 存储用户和物品的组合
    for i in range(num_user):
        sample.extend([[i,j] for j in range(num_item)])
    return np.array(sample)

# 
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x)) 

# 基于矩阵分解模型（这个 MF_BaseModel 类实现了一个基本的矩阵分解模型，利用用户和项目的嵌入表示，计算用户对项目的偏好预测。）
class MF_BaseModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF_BaseModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k) # 用户的嵌入层
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k) # 物品的嵌入层
        self.sigmoid = torch.nn.Sigmoid() # 激活函数
        self.xent_func = torch.nn.BCELoss() # 二元交叉熵损失函数（BCELoss），用于评估模型输出与实际标签之间的差异，适用于二分类问题。

    def forward(self, x, is_training=False):
        # 定义了前向传播的方法。x 是输入数据，通常是一个包含用户和项目索引的数组，
        user_idx = torch.LongTensor(x[:, 0]).cuda() # 从输入数组中提取用户索引
        item_idx = torch.LongTensor(x[:, 1]).cuda() # 从输入数组中提取物品索引
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = self.sigmoid(torch.sum(U_emb.mul(V_emb), 1)) # 逐元素相乘用户和项目的嵌入向量，在第一维度上求和，得到每对用户和项目的相似度分数。相似度分数通过 Sigmoid 函数映射到 0 到 1 之间，作为最终的预测输出。

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def predict(self, x):
        # 预测方法
        pred = self.forward(x)
        return pred.detach().cpu()        

# 基于神经网络的协同过滤基础模型（这个 NCF_BaseModel 类实现了一个基本的神经协同过滤模型，利用用户和项目的嵌入表示，结合线性层来预测用户对项目的偏好。）
class NCF_BaseModel(nn.Module):
    """The neural collaborative filtering method.
    """
    def __init__(self, num_users, num_items, embedding_k=4):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, 1) # 一个线性层，将连接的用户和项目嵌入的大小从 embedding_k*2 转换为 1，即生成一个预测值。
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        z_emb = torch.cat([U_emb, V_emb], axis=1) # 将用户和项目的嵌入向量在第二维（特征维度）上拼接，形成一个新的嵌入向量 z_emb，其大小为 embedding_k * 2

        h1 = self.linear_1(z_emb).squeeze() # 将合并后的嵌入向量通过线性层进行变换，输出一个形状为 (batch_size, 1) 的张量，去除输出张量的多余维度，使其变为一维张量，方便后续处理。
        out = self.sigmoid(h1) # 将线性层的输出通过 Sigmoid 函数，得到一个在 0 到 1 之间的概率值，表示用户对项目的偏好。
        if is_training:
            return out, z_emb
        else:
            return out    
    
    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().cpu().numpy()      

# 神经协同过滤模型
class NCF(nn.Module):

    def __init__(self, num_users, num_items, embedding_k=64):
        super(NCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, 1)
        self.relu = torch.nn.ReLU() # ReLU 激活函数
        self.sigmoid = torch.nn.Sigmoid()

        self.xent_func = torch.nn.BCELoss()


    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        
        # 将用户索引转换为嵌入向量
        U_emb = self.W(user_idx)
        
        # 将视频索引转换为嵌入向量
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)
        out = self.linear_1(z_emb).squeeze()

        if is_training:
            return self.sigmoid(out), z_emb
        else:
            return self.sigmoid(out)

    def fit(self, x, y, num_epoch=1000, lr=0.05, lamb=0, tol=1e-4, batch_size=128, verbose=1):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        # 训练模型的方法
        last_loss = 1e9 # 记录上一个周期的损失。

        num_sample = len(x)
        total_batch = num_sample // batch_size

        early_stop = 0 
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # 生成从 0 到 n-1 的整数索引数组
            np.random.shuffle(all_idx) # 打乱样本顺序。
            epoch_loss = 0
            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size] # 选中的索引数组
                sub_x = x[selected_idx]
                sub_y = torch.Tensor(y[selected_idx]).cuda()

                optimizer.zero_grad() # 清空优化器的梯度。
                
                # 前向传播
                pred = self.forward(sub_x, False)
                
                # 计算均方误差损失（MSELoss），并进行反向传播。
                xent_loss = nn.MSELoss()(pred, sub_y)
                loss = xent_loss
                loss.backward()
                
                # 更新模型参数，并累加每个批次的损失到 epoch_loss。
                optimizer.step()
                epoch_loss += xent_loss.detach().cpu().numpy()
            
            # 计算相对损失变化，如果变化小于容忍度 tol，则可能已收敛。如果连续超过 5 个周期没有显著改善，打印日志并停止训练。
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("NCF(神经协同过滤模型)] epoch:{}, xent(损失:均方误差):{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[NCF(神经协同过滤模型)] epoch:{}, xent(损失:均方误差):{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[Warning] Reach preset epochs, it seems does not converge.")

    def partial_fit(self, x, y, num_epoch=1000, lr=0.05, lamb=0, tol=1e-4):
        self.fit(x, y, num_epoch=1000, lr=0.05, lamb=0, tol=1e-4)

    def predict(self, x):
        pred, z_emb = self.forward(x, True)
        return pred.detach().cpu().numpy().flatten(), z_emb.detach().cpu().numpy()

# 矩阵分解模型。
class MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = self.sigmoid(torch.sum(U_emb.mul(V_emb), 1))

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, x, y, 
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, 
        tol=1e-4, verbose=True):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // batch_size

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # 1-6960
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_y = torch.Tensor(sub_y).cuda()

                pred, u_emb, v_emb = self.forward(sub_x, True)
                
                xent_loss = self.xent_func(pred,sub_y)

                loss = xent_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF矩阵分解模型] epoch:{}, xent(二元交叉熵损失函数):{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF矩阵分解模型] epoch:{}, xent(二元交叉熵损失函数):{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred, u_emb, v_emb = self.forward(x, True)
        z_emb = torch.cat([u_emb, v_emb], axis=1)
        return pred.detach().cpu().numpy(), z_emb.detach().cpu().numpy()

# 多层感知机
class MLP(nn.Module):
    def __init__(self, input_size, *args, **kwargs):
        super().__init__()     
        self.input_size = input_size # 输入数据的特征数量。
        self.linear_1 = torch.nn.Linear(2*self.input_size, 5) # 定义一个全连接层（线性层），其输入特征维度为 2 * input_size，输出特征维度为 5。
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()    
    
    def forward(self, x):        
        
        out = nn.Softmax(dim = 1)(self.linear_1(x).squeeze()) # 最终的输出结果，表示每个样本属于各个类别的概率。
        return out

# 我们的神经协同过滤模型
class NCF_ours(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4):
        super(NCF_ours, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.NCF_model = MF_BaseModel(num_users, num_items, embedding_k) # 创建一个基础的矩阵分解模型
        self.MLP_model = MLP(input_size = self.embedding_k) # 创建一个多层感知机模型，输入维度为 embedding_k。
        self.sigmoid = torch.nn.Sigmoid()
        self.propensity_model = NCF_BaseModel(num_users, num_items, embedding_k) # 创建一个神经协同过滤模型模型，用于处理倾向评分（propensity）
        self.xent_func = torch.nn.BCELoss() # 定义二元交叉熵损失函数（BCELoss）。
        self.mse_loss = nn.MSELoss(reduction = 'sum') # 定义均方误差损失函数，设定为“求和”模式。
    
    def fit(self, x, t, c, y, num_epoch=1000, lr=0.01, lamb=1e-4, tol=1e-4, batch_size=4096, verbose=True,
           alpha1 = 1, alpha = 1, beta = 1, theta = 1, gamma = 1, rho = 1, eta = 1, thr = 0.05):
        '''
        alpha1, alpha, beta, theta, gamma, rho, eta: 用于组合损失函数的权重。
        thr: 用于限制倾向评分的下界。
        '''
        optimizer_NCF = torch.optim.Adam(self.NCF_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_MLP = torch.optim.Adam(self.MLP_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_prop = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9
        
        num_sample = len(t)
        total_batch = num_sample // batch_size

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_t = t[selected_idx]
                sub_c = c[selected_idx]
                sub_y = y[selected_idx]
                
                # 使用倾向模型预测结果，并进行裁剪以限制范围。
                pred = self.propensity_model.forward(sub_x, False).squeeze()
                pred = torch.clip(pred, thr, 1)
                
                # 计算 CTR 损失（点击率损失），使用均方误差。
                ctr_loss = nn.MSELoss(reduction = 'sum')(pred, torch.Tensor(sub_t).cuda())
                
                # 通过 NCF 模型获取用户和项目的嵌入向量，然后拼接。
                _, pred_class_emb_u, pred_class_emb_v = self.NCF_model.forward(sub_x, True)
                pred_class_emb = torch.cat([pred_class_emb_u, pred_class_emb_v], axis=1)
                
                # 使用 MLP 模型进行类别预测。
                pred_class = self.MLP_model.forward(pred_class_emb)
                
                # 损失函数计算
                L1 = F.binary_cross_entropy(pred * pred_class[:, 0] + 1e-6, torch.Tensor(sub_t * (1-sub_c) * (1-sub_y)).cuda(), reduction = 'sum')
                L2 = F.binary_cross_entropy(pred * pred_class[:, 1] + 1e-6, torch.Tensor(sub_t * (1-sub_c) * sub_y).cuda(), reduction = 'sum')
                L3 = F.binary_cross_entropy(pred * pred_class[:, 2] + 1e-6, torch.Tensor(sub_t * sub_c * (1-sub_y)).cuda(), reduction = 'sum')
                L4 = F.binary_cross_entropy(pred * (pred_class[:, 4] + pred_class[:, 3]) + 1e-6, torch.Tensor(sub_t * sub_c * sub_y).cuda(), reduction = 'sum')
                L5 = F.binary_cross_entropy((1-pred) * (pred_class[:, 4] + pred_class[:, 1]) + 1e-6, torch.Tensor((1-sub_t) * (1-sub_c) * sub_y).cuda(), reduction = 'sum')
                L6 = F.binary_cross_entropy((1-pred) * (pred_class[:, 0] + pred_class[:, 2] + pred_class[:, 3]) + 1e-6, torch.Tensor((1-sub_t) * (1-sub_c) * (1-sub_y)).cuda(), reduction = 'sum')   
                
                loss = (alpha1 * L1 + alpha * L2 + beta * L3 + theta * L4 + gamma * L5 + rho * L6) + eta * ctr_loss
                
                # 反向传播与参数更新
                optimizer_NCF.zero_grad()
                optimizer_MLP.zero_grad()
                optimizer_prop.zero_grad()
                loss.backward()                
                optimizer_NCF.step()
                optimizer_MLP.step()
                optimizer_prop.step()
                
                epoch_loss += loss.detach().detach().cpu().numpy()
            
            # 计算相对损失变化以判断收敛情况，如果变化小于阈值且连续多次未改进，则触发早停。
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[NCF_ours] epoch:{}, xent(复合损失函数):{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[NCF_ours] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[Warning] Reach preset epochs, it seems does not converge.")
    
    def predict(self, x):
        _, pred_class_emb_u, pred_class_emb_v = self.NCF_model.forward(x, True)
        pred_class_emb = torch.cat([pred_class_emb_u, pred_class_emb_v], axis=1)
        pred_class = self.MLP_model.forward(pred_class_emb)
        return pred_class.detach().cpu().numpy().flatten() 
    
def one_hot(x):
    # 将输入的二进制数组（0 和 1）转换为 one-hot 编码格式。
    out = torch.cat([torch.unsqueeze(1-x,1),torch.unsqueeze(x,1)],axis=1)
    return out

def sharpen(x, T):
    # 通过对输入的张量进行指数缩放，增强或减弱概率分布的峰度（sharpness）。
    temp = x**(1/T)
    return temp / temp.sum(1, keepdim=True)
