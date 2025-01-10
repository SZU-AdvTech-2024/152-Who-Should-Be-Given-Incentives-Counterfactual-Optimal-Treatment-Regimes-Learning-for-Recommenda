# -*- coding: utf-8 -*-
import numpy as np

# defaultdict 是一个字典的子类，提供了一个默认值的功能，允许你在访问一个不存在的键时，不会抛出 KeyError，而是返回一个默认值。
from collections import defaultdict


# environmental setting
NUM_USER = 10
NUM_ITEM = 1000


"Build several tools for simulation."

# LoggingSystem用于记录和分析用户与物品交互的日志
class LoggingSystem:
    def __init__(self, num_user=NUM_USER, num_item=NUM_ITEM):
        self.num_user = num_user
        self.num_item = num_item
        self.items_ = np.zeros(num_item) # 创建一个大小为 num_item 的 NumPy 数组，初始值全为 0，用于记录每个物品的状态或统计信息。
        self.sample_ = None # 初始化 self.sample_ 为 None，这个变量将用于存储用户交互的日志数据。

    def update(self, logging):
        '''
        更新日志
        参数 logging 是一个数组，包含用户交互的日志信息。
        '''
        self.items_ += np.bincount(logging[:,
                                   1].astype(int), minlength=self.num_item) # 使用 np.bincount 统计 logging 中第二列（即物品 ID）的出现次数，并将这些计数加到 self.items_ 中。minlength=self.num_item 确保计数数组的长度与物品数量一致。
        if self.sample_ is None:
            self.sample_ = logging
        else:
            self.sample_ = np.concatenate([self.sample_, logging], axis=0)

    @property
    def user_wise_ctr(self):
        # 用户点击率 (CTR) 计算
        ctr_list = [] # ctr_list 用于存储每个用户的 CTR。
        for i in range(self.num_user):
            user_sample = self.sample_[self.sample_[:, 0] == i]
            ctr_list.append(user_sample[:, -1].mean()) # 计算该用户最后一列的平均值，并将其添加到 ctr_list 中。
        return np.array(ctr_list)

    @property
    def user_wise_ctr_last(self):
        # 计算最近的用户点击率
        # last 100 loggings ctr
        ctr_list = []
        for i in range(self.num_user):
            user_sample = self.sample_[self.sample_[:, 0] == i]
            ctr_list.append(user_sample[-10:, -1].mean()) # 选择最近 10 条交互记录，并计算其点击率的平均值。
        return np.array(ctr_list)

    @property
    def item_wise_ctr(self):
        # 物品点击率计算
        ctr_list = []
        for i in range(self.num_item):
            item_sample = self.sample_[self.sample_[:, 1] == i]
            ctr_list.append(item_sample[:, -1].mean().astype(np.float32)) 
        # 计算每个物品的平均点击率。
        return np.array(ctr_list)

    @property
    def item_stat(self):
        #  物品统计信息
        return self.items_.astype(int)

    @property
    def logs(self):
        # 日志信息
        return self.sample_

    @property
    def logs_last(self):
        # 最近的日志信息
        return self.sample_[-100:]


class Policy:
    """A policy to assign items to each user.
    """
    # 用于为用户分配物品并预测点击率（CTR）
    def __init__(self, model, num_user, num_item):
        # initialize a random policy
        self.policy = None # 初始化 self.policy 为 None，表示尚未学习到具体的策略，最初的策略为随机。
        # initialize the model
        self.model = model
        self.num_user = num_user
        self.num_item = num_item

    def learn(self, x, y, *args, **kwargs):
        # 负责学习推荐策略
        # learn the policy
        if self.policy is None:
            """None indicates random policy.
            """
            self.policy = self.model

            if self.policy is not None:
                if kwargs["lamb"] is not None:
                    self.policy.fit(x, y, lamb=kwargs["lamb"])
                else:
                    # LR method
                    self.policy.fit(x, y)

        else:
            """Given MF(矩阵分解模型) or MF-CRRM(矩阵分解的扩散模型) model.
            """
            if kwargs["lamb"] is not None:
                self.policy.fit(x, y, lamb=kwargs["lamb"])
            else:
                self.policy.fit(x, y)

            return

    def predict(self, x_test):
        # predict the ctr rate
        # 预测点击率
        if self.policy is None:
            return np.ones(x_test.shape[0]) * 0.5
        else:
            return self.policy.predict_proba(x_test)[:, 1]

    def forward(self, user_idx, top_n=3):
        """Give user idx, gives recommended items.
        推荐物品
        """
        if self.policy is None:
            # give random assignment
            pred = np.random.rand(len(user_idx), self.num_item)
        else:
            item_idx = np.arange(self.num_item)
            sample = [] # 所有用户-物品对的列表
            # generate test samples
            for user in user_idx:
                sample.extend([[user, item] for item in item_idx])

            x_test = np.array(sample)

            pred = self.policy.predict_proba(x_test)[:, 1]
            # rank item
            pred = pred.reshape(self.num_user, self.num_item)

        # deterministically rank from top to end
        pred_item = np.argsort(-pred, 1)[:, :top_n]

        return pred_item


def simulate_click_or_not(pred_item, USER_PREF_MAT):
    """Given the predicted item index,
    simulate the user's click or not based on
    the given USER_PREF_MAT.
    """
    '''
    pred_item: 预测的物品索引，通常是用户可能点击的物品列表。
    USER_PREF_MAT: 用户偏好矩阵，包含用户对各物品的真实偏好值。
    '''
    sample = [] # 用于存储每个用户、物品和点击标签的三元组。

    for user in range(USER_PREF_MAT.shape[0]):
        real_pref = USER_PREF_MAT[user][pred_item[user]] # 获取当前用户对其预测物品的偏好值：
        response = np.zeros_like(real_pref) # 存储每个预测物品的点击响应
        # click
        response[np.random.rand(real_pref.shape[0]) < real_pref] = 1
        response_b = response.astype(bool)

        sample += [[user, item, label]
                   for item, label in zip(pred_item[user], response)]

    sample_ar = np.array(sample)
    return sample_ar[:, :-1], sample_ar[:, -1] # 返回用户和物品的索引以及点击标签


def generate_total_sample(num_user, num_item):
    # 生成一个包含所有用户与所有物品的组合的样本
    sample = []
    for i in range(num_user):
        sample.extend([[i, j] for j in range(num_item)])
    return np.array(sample)


def gini_index(user_utility):
    # 计算基尼指数
    # user_utility: 这是一个数组或列表，表示每个用户的效用值（utility），通常用于衡量用户对某些物品的满意度或偏好。
    from sklearn.metrics import auc
    # auc 用于计算曲线下的面积（Area Under Curve），在这里用于计算洛伦兹曲线下的面积。
    cum_L = np.cumsum(np.sort(user_utility)) # 计算排序后效用值的累积和。cum_L[i] 表示前 i 个用户的效用值的总和。
    sum_L = np.sum(user_utility) # user_utility 计算基尼指数时需要的总效用值。
    num_user = len(user_utility)
    xx = np.linspace(0, 1, num_user + 1) # xx表示用户的比例。
    yy = np.append([0], cum_L / sum_L) # yy 数组表示洛伦兹曲线的纵坐标。

    gi = (0.5 - auc(xx, yy)) / 0.5 # gi 代表基尼指数，表示不平等的程度
    gu = sum_L / num_user # gu 表示每个用户的平均效用。

    print("Num User:", num_user)
    print("Gini index:", gi)
    print("Global utility:", gu)
    return gi, gu


def rating_mat_to_sample(mat):
    # 将评分矩阵转换为样本
    # mat 通常表示用户对物品的评分或效用值
    row, col = np.nonzero(mat) # row 和 col。这两个数组分别包含矩阵 mat 中所有非零元素的行索引和列索引
    y = mat[row, col] # y 数组包含所有评分的值，形状为 (n,)，其中 n 是非零评分的数量。
    x = np.concatenate([row.reshape(-1, 1), col.reshape(-1, 1)], axis=1)
    return x, y # x: 代表用户与物品的索引配对（每行是 [user_index, item_index]）。y: 代表相应的评分值（这些评分是与 x 中的用户和物品索引对应的）。


def delete_negative_sample(x, y):
    # 从给定的样本中筛选出正样本
    positive_idx = (y == 1) # positive_idx 数组的长度与 y 相同，且其中的 True 值表示正样本的索引。
    return(x[positive_idx], y[positive_idx])


def binarize(y, thres=3):
    # 用户评分数组 y 根据指定的阈值进行二值化。
    """Given threshold, binarize the ratings.
    """
    # thres：二值化的阈值
    y[y < thres] = 0
    y[y >= thres] = 1
    return y


def shuffle(x, y):
    # 将输入数据 x 和对应的标签 y 同时随机打乱。
    idxs = np.arange(x.shape[0])
    np.random.shuffle(idxs)
    return x[idxs], y[idxs]


def get_user_wise_ctr(x_test, y_test, test_pred, top_N=5):
    offset = 0 # 初始化一个偏移量 offset，它将用于调整索引，以便正确获取物品的索引。
    user_idxs = np.unique(x_test[:, 0])
    user_ctr_list = [] # 存储每个用户的点击率（CTR）。
    for user in user_idxs:
        mask = x_test[:, 0] == user
        pred_item = np.argsort(-test_pred[mask])[:top_N] + offset
        u_ctr = y_test[pred_item].sum() / pred_item.shape[0] # 当前用户的点击率（CTR）
        user_ctr_list.append(u_ctr)
        offset += mask.sum()

    user_ctr_list = np.array(user_ctr_list)
    return user_ctr_list


def minU(x_test, y_test, test_pred, top_N=5):
    # 计算每个用户在推荐系统中的点击率（CTR，Click-Through Rate），并找出最小的点击率及其出现次数。
    offset = 0
    user_idxs = np.unique(x_test[:, 0])
    user_ctr_list = []
    for user in user_idxs:
        mask = x_test[:, 0] == user
        pred_item = np.argsort(-test_pred[mask])[:top_N] + offset
        u_ctr = y_test[pred_item].sum() / pred_item.shape[0]
        user_ctr_list.append(u_ctr)
        offset += mask.sum()

    user_ctr_list = np.array(user_ctr_list)
    print("minU最小点击率: {}, # of minU最小点击率出现的次数: {}".format(
        min(user_ctr_list), sum(user_ctr_list == min(user_ctr_list))))
    return user_ctr_list


def generate_rcts(num_sample, USER_PREF_MAT):
    # 生成随机的“推荐-选择-反馈”样本（RCTs），用于模拟用户对物品的偏好
    num_user = USER_PREF_MAT.shape[0]
    num_item = USER_PREF_MAT.shape[1]
    
    # 随机选择用户和物品索引
    idx1 = np.random.randint(0, num_user, num_sample)
    idx2 = np.random.randint(0, num_item, num_sample)
    
    # 生成预测偏好矩阵
    pred = np.random.rand(USER_PREF_MAT.shape[0], USER_PREF_MAT.shape[1])
    pred_item = np.argsort(-pred, 1)[:, :3]
    
    # 生成反馈标签
    mask = pred[idx1, idx2] > USER_PREF_MAT[idx1, idx2]
    y = mask.astype(int)
    
    # 构建特征（x）
    x = np.concatenate([idx1.reshape(-1, 1),
                        idx2.reshape(-1, 1)], axis=1)

    return x, y


def ndcg_func(model, x_te, y_te, top_k_list=[5, 10]):
    """Evaluate nDCG@K of the trained model on test dataset.
    """
    all_user_idx = np.unique(x_te[:, 0])
    all_tr_idx = np.arange(len(x_te))
    result_map = defaultdict(list)

    for uid in all_user_idx:
        u_idx = all_tr_idx[x_te[:, 0] == uid]
        x_u = x_te[u_idx]
        y_u = y_te[u_idx]
        pred_u = model.predict(x_u)

        for top_k in top_k_list:
            pred_top_k = np.argsort(-pred_u)[:top_k]
            count = y_u[pred_top_k].sum()

            log2_iplus1 = np.log2(1+np.arange(1, top_k+1))

            dcg_k = y_u[pred_top_k] / log2_iplus1

            best_dcg_k = y_u[np.argsort(-y_u)][:top_k] / log2_iplus1

            if np.sum(best_dcg_k) == 0:
                ndcg_k = 1
            else:
                ndcg_k = np.sum(dcg_k) / np.sum(best_dcg_k)

            result_map["ndcg_{}".format(top_k)].append(ndcg_k)

    return result_map


def relative_item_popularity(x, num_user, num_item, eta=0.5):
    full_matrix = np.zeros((num_user, num_item))
    for pair in x:
        full_matrix[pair[0], pair[1]] = 1
    item_popularity = np.sum(full_matrix, axis=0).reshape(-1)
    max_popularity = max(item_popularity)
    item_popularity[item_popularity == 0] = 1
    item_popularity /= max_popularity
    return item_popularity**eta
