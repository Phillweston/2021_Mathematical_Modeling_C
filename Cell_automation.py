# This is a cell automation project
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans


class GameOfLife(object):

    def __init__(self, cells_shape):
        """
        Parameters
        ----------
        cells_shape : 一个元组，表示画布的大小。

        Examples
        --------
        建立一个高20，宽30的画布
        game = GameOfLife((20, 30))

        """

        # 矩阵的四周不参与运算
        self.cells = np.zeros(cells_shape)
        print(cells_shape)
        # 实际参与运算的矩阵的宽和高，上下左右四条边不算进去，所以要-2
        real_width = cells_shape[0] - 2
        real_height = cells_shape[1] - 2
        print(real_width)
        # 初始化细胞的位置，生成real_width*real_height的随机矩阵，其中参数大于等于2
        self.cells[1:-1, 1:-1] = np.random.randint(2, size=(real_width, real_height))
        self.position = np.zeros([real_width * real_height, 2])
        self.timer = 0
        self.mask = np.ones(9)
        # self.mask[4]代表九宫格最中间的那个元素，也就是自身
        self.mask[4] = 0
        self.count = 0
        # 预测结果
        self.label_predict = []
        # 预测的类别数
        self.region_number = 6
        # 每个类别的统计数据
        self.region_statistics = np.zeros(self.region_number)
        # 预测的聚类中心
        self.centroids = []
        # 各个标签对应的数据点
        self.cluster_k = []

    def cluster(self):
        """k_means聚类分析"""
        self.region_statistics = np.zeros(self.region_number)
        position = self.position
        # 将生成的数据的聚类分为6类
        estimator = KMeans(n_clusters=self.region_number)
        # 拟合+预测
        # 注意这里的要传入的参数是修改后的n*2数组，第一维度存放的是横坐标，第二维度存放的是纵坐标
        # res代表返回的聚类结果，范围是从0到类别数-1
        res = estimator.fit_predict(self.position)
        print("Kmeans result is %s" % res)
        # 预测类别标签结果
        self.label_predict = estimator.labels_
        # 各个类别的聚类中心值
        self.centroids = estimator.cluster_centers_
        inertia = estimator.inertia_
        # 各个标签对应的数据点
        self.cluster_k = [self.position[res == k] for k in range(self.region_number)]
        # print(self.cluster_k)
        # print(self.label_predict)
        '''
        采用DBI的方式求解
        '''
        # 求簇类中数据到质心欧氏距离和的平均距离
        S = [np.mean([euclidean(p, self.centroids[i]) for p in k]) for i, k in enumerate(self.cluster_k)]
        # 求出Rij和Ri:
        Ri = []
        for i in range(self.region_number):
            Rij = []
            # 衡量第i类与第j类的相似度
            for j in range(self.region_number):
                if j != i:
                    r = (S[i] + S[j]) / euclidean(self.centroids[i], self.centroids[j])  # 这个分母是Mij，即两个质心的距离
                    Rij.append(r)
            Ri.append(max(Rij))
        # 求DBI值
        dbi = np.mean(Ri)
        print("DBI result is %s" % dbi)

    def update_state(self):
        """更新一次状态"""
        buf = np.zeros(self.cells.shape)
        cells = self.cells
        self.count = 0
        # 注意这里要从1开始到最后一个值减一
        for i in range(1, cells.shape[0] - 1):
            for j in range(1, cells.shape[1] - 1):
                # 计算该细胞周围的存活细胞数
                # 一个元胞的生死由其在该时刻本身的生死状态和周围的八个邻居的状态
                neighbor = cells[i - 1:i + 2, j - 1:j + 2].reshape((-1,))
                # 线性卷积函数，self.mask*neighbor
                # 返回的数组长度为max(M,N)-min(M,N)+1,此时返回的是完全重叠的点。边缘的点无效。
                neighbor_num = np.convolve(self.mask, neighbor, 'valid')[0]
                if neighbor_num == 3:
                    # 如果这个元胞周围有三个元胞为生，则这个元胞为生
                    buf[i, j] = 1
                elif neighbor_num == 2:
                    # 如果这个元胞周围有两个元胞为生，则这个元胞保持之前的状态
                    buf[i, j] = cells[i, j]
                else:
                    # 如果这个元胞周围有一个或零个元胞为生，则这个元胞为死
                    buf[i, j] = 0
                # 如果这个细胞为生，则加入到聚类中
                if buf[i, j] == 1:
                    # 为什么要进行这一步？
                    # 需要将目前存活的细胞的位置加入到n*2的数组中
                    self.position[self.count][0] = i
                    self.position[self.count][1] = j
                    self.count += 1
        self.cells = buf
        self.timer += 1

    def plot_state(self):
        """画出当前的状态"""
        label_predict = self.label_predict
        position = self.position
        region_statistics = self.region_statistics
        centroids = self.centroids
        # 显示第一张图：细胞自动机图示
        plt.subplot(2, 2, 1)
        plt.title('Cell Automation Iter :{}'.format(self.timer))
        plt.imshow(self.cells, origin='lower')
        # 注意这里要从1开始到最后一个值减一，循环self.count次数（原因：构造的self.position中未赋值的变量都是0，只能循环遍历赋值后的变量）
        # 显示第二张图：聚类分析图示
        plt.subplot(2, 2, 2)
        for i in range(0, self.count):
            if int(label_predict[i]) == 0:
                plt.scatter(position[i][0], position[i][1], color='red')
                region_statistics[0] += 1
            elif int(label_predict[i]) == 1:
                plt.scatter(position[i][0], position[i][1], color='black')
                region_statistics[1] += 1
            elif int(label_predict[i]) == 2:
                plt.scatter(position[i][0], position[i][1], color='blue')
                region_statistics[2] += 1
            elif int(label_predict[i]) == 3:
                plt.scatter(position[i][0], position[i][1], color='green')
                region_statistics[3] += 1
            elif int(label_predict[i]) == 4:
                plt.scatter(position[i][0], position[i][1], color='yellow')
                region_statistics[4] += 1
            elif int(label_predict[i]) == 5:
                plt.scatter(position[i][0], position[i][1], color='brown')
                region_statistics[5] += 1
        plt.title('Cluster Iter :{}'.format(self.timer))
        plt.legend(['A', 'B', 'C', 'D', 'E', 'F'])
        # 显示第三张图：地区统计数据
        plt.subplot(2, 2, 3)
        plt.bar(range(self.region_number), region_statistics, align='center', color='steelblue', alpha=0.8)
        plt.ylabel('statistic_number')
        plt.xticks(range(self.region_number), ['A', 'B', 'C', 'D', 'E', 'F'])
        plt.ylim([min(region_statistics), max(region_statistics)])
        for x, y in enumerate(region_statistics):
            plt.text(x, y, '%s' % y, ha='center')
        # 显示第四张图，聚类中心位置
        plt.subplot(2, 2, 4)
        for i in range(self.region_number):
            if i == 0:
                plt.scatter(centroids[i][0], centroids[i][1], color='red')
            elif i == 1:
                plt.scatter(centroids[i][0], centroids[i][1], color='black')
            elif i == 2:
                plt.scatter(centroids[i][0], centroids[i][1], color='blue')
            elif i == 3:
                plt.scatter(centroids[i][0], centroids[i][1], color='green')
            elif i == 4:
                plt.scatter(centroids[i][0], centroids[i][1], color='yellow')
            elif i == 5:
                plt.scatter(centroids[i][0], centroids[i][1], color='brown')
        plt.title('Cluster Center Iter :{}'.format(self.timer))
        plt.show()

    def update_and_plot(self, n_iter):
        """更新状态并画图
        Parameters
        ----------
        n_iter : 更新的轮数
        """
        plt.ion()
        for _ in range(n_iter):
            plt.title('Iter :{}'.format(self.timer))
            self.update_state()
            self.cluster()
            self.plot_state()
            # plt.pause(0.2)
        plt.ioff()


if __name__ == '__main__':
    game = GameOfLife(cells_shape=(60, 60))
    game.update_and_plot(200)
