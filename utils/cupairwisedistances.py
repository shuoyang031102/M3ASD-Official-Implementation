

import time
import tensorflow as tf
import cupy as np


def get_distance(vector1, vector2):
    num1 = vector1 - np.average(vector1)
    num2 = vector2 - np.average(vector2)
    num = np.sum(num1 * num2)
    den = np.sqrt(np.sum(np.power(num1,2)) * np.sum(np.power(num2,2)))
    if den == 0:
        return 0.0
    return np.abs(num/den)


def get_pairwise_distances(embeddings):
    tf.compat.v1.disable_eager_execution()
    """
        计算嵌入向量之间的皮尔逊相关系数
        Args:
            embeddings: 形如(batch_size, embed_dim)的张量
        Returns:
            piarwise_distances: 形如(batch_size, batch_size)的张量
    """
    avg_vec = tf.reduce_mean(embeddings, axis=1)
    # 归一到期望E(x)=0
    nomal_embed = embeddings - tf.expand_dims(avg_vec, 1)

    # 计算 sum((x-avg(x))*(y-avg(y)))的混淆矩阵，即分子矩阵
    dot_product = tf.matmul(nomal_embed, tf.transpose(nomal_embed))

    # 计算分母 sqrt((x-avg(x))^2 * (y-avg(y))^2)
    square_norm = tf.compat.v1.diag_part(dot_product)
    square_norm = tf.matmul(tf.expand_dims(square_norm, 1), tf.expand_dims(square_norm, 0))

    distance = dot_product / tf.sqrt(square_norm)

    return tf.compat.v1.Session().run(distance)

if __name__ == "__main__":

    a = np.array([[2, 7, 18, 88, 157, 90, 177, 570],
                  [3, 5, 15, 90, 180, 88, 160, 580],
                  [1, 2, 3, 4, 5, 6, 7, 8]], dtype=float)
    print(a.shape)
    start = time.time()
    dis = []
    for i in range(a.shape[0] - 1):
        for j in range(i + 1, a.shape[0]):
            dis.append(get_distance(a[i, :], a[j, :]))
    end = time.time()
    for i in dis:
        print(i)
    print('for cost {0}s'.format(end - start))

    start = time.time()
    dis = get_pairwise_distances(a)
    end = time.time()
    print(dis)
    print('matrix cost {0}s'.format(end - start))
