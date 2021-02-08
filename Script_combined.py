# This is a script combined project
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import requests
import urllib.parse
import urllib.request
import hashlib
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from pandas.core.frame import DataFrame

'''
Latitude_and_Longitude_Array 各点在坐标的位置
Population_Point 种群点坐标
Plot_x_range x轴坐标范围
Plot_y_range y轴坐标范围
Point_count 种群点数量
'''
# 读取2019年的经纬度以及Lab Status数据
Latitude_and_Longitude_Data = pd.read_excel('Proceed_Data2020.xlsx')
# 读取经纬度掩膜信息（掩膜信息为1说明该地区数据有效，掩膜信息为0说明该地区数据无效）
Latitude_and_Longitude_Mask_Data = pd.read_excel('Proceed_Data_Mask.xlsx')
# 用于存放经纬度转化之后的数据以及对应的Positive判断值
Latitude_and_Longitude_Array = np.zeros([Latitude_and_Longitude_Data.shape[0], 3])
# 用于存放经纬度掩膜转化之后的数据以及对应的Available判断值
Latitude_and_Longitude_Mask_Array = np.zeros([Latitude_and_Longitude_Mask_Data.shape[0], 3])
# 设置经纬度最大最小值
Latitude_max = Latitude_and_Longitude_Data['Latitude'].max()
Latitude_min = Latitude_and_Longitude_Data['Latitude'].min()
Longitude_max = Latitude_and_Longitude_Data['Longitude'].max()
Longitude_min = Latitude_and_Longitude_Data['Longitude'].min()
# 定义经纬度比例尺（即一度代表多少km）
Latitude_scale = 111
Longitude_scale = 111 * math.cos(Latitude_scale)
# 定义绘图坐标系的范围
Plot_scale = 2
Plot_x_range = abs(round((Latitude_max - Latitude_min) * Latitude_scale / Plot_scale))
Plot_y_range = abs(round((Longitude_max - Longitude_min) * Longitude_scale / Plot_scale))

Population_Number = (Latitude_and_Longitude_Data['Lab Status']).tolist().count('Positive ID')
# 统计各个已知的种群点
Population_Point = np.zeros([Population_Number, 2])
Point_count = 0
# 经纬度信息转换到绘图坐标系上面
for i in range(Latitude_and_Longitude_Data.shape[0]):
    Latitude_and_Longitude_Array[i][0] = round(
        float(Latitude_and_Longitude_Data.loc[i, ['Latitude']] - Latitude_min) * Latitude_scale / Plot_scale)
    Latitude_and_Longitude_Array[i][1] = -round(
        float(Latitude_and_Longitude_Data.loc[i, ['Longitude']] - Longitude_min) * Longitude_scale / Plot_scale)
    if (Latitude_and_Longitude_Data.loc[i, ['Lab Status']]).tolist() == ['Positive ID']:
        Latitude_and_Longitude_Array[i][2] = 1
        # 去掉分离较远的种群点
        if (Population_Point[Point_count][0] > 180 and Population_Point[Point_count][1] < 15) == 0:
            Population_Point[Point_count][0] = Latitude_and_Longitude_Array[i][0]
            Population_Point[Point_count][1] = Latitude_and_Longitude_Array[i][1]
            Point_count += 1
    else:
        Latitude_and_Longitude_Array[i][2] = 0
# print(Population_Point)

print("Please wait while the program is calculating cell mask")
# 经纬度掩膜信息转换到绘图坐标系上面
for i in range(Latitude_and_Longitude_Mask_Data.shape[0]):
    Latitude_and_Longitude_Mask_Array[i][0] = round(
        float(Latitude_and_Longitude_Mask_Data.loc[i, ['Latitude']] - Latitude_min) * Latitude_scale / Plot_scale)
    Latitude_and_Longitude_Mask_Array[i][1] = -round(
        float(Latitude_and_Longitude_Mask_Data.loc[i, ['Longitude']] - Longitude_min) * Longitude_scale / Plot_scale)
    Latitude_and_Longitude_Mask_Array[i][2] = int(Latitude_and_Longitude_Mask_Data.loc[i, ['Available']])
'''
# 定义百度地图API相关参数
ak = "XGa3PomYByhWqSpHGxXfwuOFP5vS1Cmy"
sk = "SeoETux1exdFGeNLaTqPuXYAdvlYBeVu"
url = "http://api.map.baidu.com"
'''


class GameOfLife(object):

    def __init__(self, cells_shape, Point_count):
        """
        Parameters
        ----------
        cells_shape : 一个元组，表示画布的大小。

        Examples
        --------
        建立一个宽20，高30的画布
        row = 20, col = 30
        game = GameOfLife((20, 30))

        """
        cell_extend_size = 10
        # 实际参与运算的矩阵的宽和高，上下左右四条边不算进去，所以要-2
        # The width and height of the matrix actually involved in the operation, the upper and lower left and right sides do not count in, so to -2
        # 实际的矩阵会赋值经纬度转化对应点，对应的cell_shape即为从第1个元素开始赋值一直到第end-1个元素结束
        # The actual matrix assigns a longitude and latitude conversion corresponding point, and the corresponding cell_shape is assigned from the 1st element to the end of the 1st element
        real_width = cells_shape[0] - 2
        real_height = cells_shape[1] - 2
        # print(real_width, real_height)
        # 定义细胞初始状态，矩阵的四周不参与运算
        # Defines the initial state of the cell, and the four persetrics of the matrix are not involved in the operation
        self.cells = np.zeros(cells_shape)
        # print(cells_shape)
        # 细胞的卷积移动平均值
        # The average of the cell's co product movement
        self.cell_value = np.zeros([cells_shape[0], cells_shape[1]])
        # 定义细胞掩膜初始状态，矩阵的四周不参与运算
        # Defines the initial state of the cell mask, and the peroth of the matrix is not involved in the operation
        self.cells_mask = np.ones(cells_shape)
        for k in range(len(Latitude_and_Longitude_Mask_Array)):
            if Latitude_and_Longitude_Mask_Array[k][2] == 0:
                self.cells_mask[int(Latitude_and_Longitude_Mask_Array[k][0]), int(Latitude_and_Longitude_Mask_Array[k][1])] = 0
        # 初始化细胞的位置，生成real_width*real_height的随机矩阵，然后添加已知的种群点坐标
        # Initialize the location of the cells, real_width a real_height matrix of the cells, and then add known population point coordinates
        # 添加规则：种群点中心位置为确定值1，点距离种群点中心越远则为1的可能性越小，包围的5*5区域形成一个簇
        # Add rule: The center of the population point is a defined value of 1, and the farther away the point is from the center of the population point, the less likely it is that 1 will form a cluster in the surrounding 5x5 area
        # 用簇的分布来近似代替种群点附近的分布
        # The distribution of clusters is used to approximate the distribution near population points
        for k in range(Point_count):
            self.cells[int(Population_Point[k][0]), int(Population_Point[k][1])] = 1
            # print(Population_Point[k][0], Population_Point[k][1])
            for i in range(-cell_extend_size, cell_extend_size):
                for j in range(-cell_extend_size, cell_extend_size):
                    # 添加这个判断条件防止参数越界
                    if 0 <= int(Population_Point[k][0]) + i < cells_shape[0] and 0 <= int(Population_Point[k][1]) + j < \
                            cells_shape[1]:
                        Probability = -0.0636 * math.sqrt(i ** 2 + j ** 2) + 1
                        self.cells[int(Population_Point[k][0]) + i, int(Population_Point[k][1]) + j] = np.random.choice(
                            [0, 1], p=[1 - Probability, Probability])

        self.position = np.zeros([real_width * real_height, 2])
        self.timer = 0
        # 定义卷积掩膜
        # Define the constumal mask
        self.mask = np.ones(9)
        # self.mask[4]代表九宫格最中间的那个元素，也就是自身
        # self.mask[4] represents the middle element of the nine palace grids, that is, itself
        self.mask[4] = 0
        self.mask_density = np.ones(25)
        self.mask_density[12] = 0
        self.count = 0
        # 预测结果
        # Predict the results
        self.label_predict = []
        # 预测的类别数
        # The number of categories predicted
        self.region_number = Point_count
        # 每个类别的统计数据
        # Statistics for each category
        self.region_statistics = np.zeros(self.region_number)
        # 预测的聚类中心
        # The clustering center of the prediction
        self.centroids = []
        # 各个标签对应的数据点
        # The data points for each label
        self.cluster_k = []
        # 定义密度绘制坐标系经纬度
        # Define density to plot the latitude and longitude of the coordinate system
        self.Density_Plot_Latitude_List = []
        self.Density_Plot_Longitude_List = []

    def cluster(self):
        """k_means聚类分析"""
        self.region_statistics = np.zeros(self.region_number)
        position = self.position
        # 将生成的数据的聚类分为Point_count类
        # Cluster the resulting data into Point_count categories
        estimator = KMeans(n_clusters=Point_count)
        # 拟合+预测
        # Fitting and forecasting
        # 注意这里的要传入的参数是修改后的n*2数组
        # Note that the argument to be passed in here is the modified array of n*2
        # res代表返回的聚类结果，范围是从0到类别数-1
        # Res represents the returned cluster result, ranging from 0 to the number of categories -1
        res = estimator.fit_predict(self.position)
        # 预测类别标签结果
        # Predict category label results
        self.label_predict = estimator.labels_
        # 各个类别的聚类中心值
        # Cluster center values for each category
        self.centroids = estimator.cluster_centers_
        inertia = estimator.inertia_
        # print(self.label_predict)
        # 各个标签对应的数据点
        # The data points for each label
        self.cluster_k = [self.position[res == k] for k in range(self.region_number)]
        # print(self.cluster_k)
        '''
        采用DBI的方式，对KMeans聚类的结果进行评估
        The results of KMeans clustering are evaluated by DBI
        '''
        # 求簇类中数据到质心欧氏距离和的平均距离
        # Find the distance between the data and the average distance from the centrity eutony in the cluster class
        S = [np.mean([euclidean(p, self.centroids[i]) for p in k]) for i, k in enumerate(self.cluster_k)]
        # 求出Rij和Ri:
        # Find Rij and Ri
        Ri = []
        for i in range(self.region_number):
            Rij = []
            # 衡量第i类与第j类的相似度
            # Measure the similarity between class i and class j
            for j in range(self.region_number):
                if j != i:
                    r = (S[i] + S[j]) / euclidean(self.centroids[i], self.centroids[j])  # 这个分母是Mij，即两个质心的距离
                    Rij.append(r)
            Ri.append(max(Rij))
        # 求DBI值
        # Find the DBI value
        dbi = np.mean(Ri)
        print("DBI result is %s" % dbi)

    def update_state(self):
        """更新一次状态"""
        buf = np.zeros(self.cells.shape)
        cells = self.cells
        cells_mask = self.cells_mask
        self.count = 0
        # 注意这里要从1开始到最后一个值减一
        for i in range(1, cells.shape[0] - 1):
            for j in range(1, cells.shape[1] - 1):
                # 计算该细胞周围的存活细胞数
                # The number of surviving cells around the cell is calculated
                # 一个元胞的生死由其在该时刻本身的生死状态和周围的八个邻居的状态
                # The life and death of a cell is determined by its own state of life and death at that moment and the state of the eight neighbors around it
                neighbor = cells[i - 1:i + 2, j - 1:j + 2].reshape((-1,))
                # 线性卷积函数，self.mask*neighbor
                # Linear co product function, self.mask*neighbor
                # 返回的数组长度为max(M,N)-min(M,N)+1,此时返回的是完全重叠的点。边缘的点无效。
                # The returned array length is max (M, N) -min (M, N) plus 1, at which point is completely overlapping. The point at the edge is not valid.
                neighbor_num = np.convolve(self.mask, neighbor, 'valid')[0]
                # 这里添加了对于当前位置的细胞掩膜的判断，掩膜值为1说明该位置细胞可能存活，掩膜为0说明该位置细胞不可能存活
                # A judgment of the cell mask at the current location is added here, with a mask value of 1 indicating that the cell may survive at that location and a mask of 0 indicating that the cell at that location is unlikely to survive
                if neighbor_num == 3 and cells_mask[i, j] != 0:
                    # 如果这个元胞周围有三个元胞为生，则这个元胞为生
                    # If there are three cells around the cell for a living, the cell is for a living
                    buf[i, j] = 1
                elif neighbor_num == 2 and cells_mask[i, j] != 0:
                    # 如果这个元胞周围有两个元胞为生，则这个元胞保持之前的状态
                    # If there are two cells around the cell for a living, the cell remains in its previous state
                    buf[i, j] = cells[i, j]
                else:
                    # 如果这个元胞周围有一个或零个元胞为生，则这个元胞为死
                    # If there is one or zero cells around the cell for a living, the cell is dead
                    buf[i, j] = 0
                # 如果这个细胞为生，则加入到聚类中
                # If the cell is raw, it is added to the cluster
                if buf[i, j] == 1:
                    # 为什么要进行这一步？
                    # 需要将目前存活的细胞的位置加入到n*2的数组中
                    # The location of the cells that are currently alive needs to be added to the array of n*2
                    self.position[self.count][0] = i
                    self.position[self.count][1] = j
                    self.count += 1

        self.cells = buf
        self.timer += 1

    def plot_state(self):
        """画出当前的状态"""
        cell_value = self.cell_value
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
        # 如果存在(0, 0)坐标点有带颜色的点，说明聚类中心存在重复坐标
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
        plt.title('Cluster Iter :{}'.format(self.timer))
        '''
        # 显示第三张图：地区统计数据
        plt.subplot(2, 2, 3)
        plt.bar(range(self.region_number), region_statistics, align='center', color='steelblue', alpha=0.8)
        plt.ylabel('statistic_number')
        plt.xticks(range(self.region_number), ['A', 'B', 'C', 'D', 'E', 'F'])
        plt.ylim([min(region_statistics), max(region_statistics)])
        for x, y in enumerate(region_statistics):
            plt.text(x, y, '%s' % y, ha='center')
        '''
        # 显示第三张图：密度分布图
        plt.subplot(2, 2, 3)
        # print(cell_value.shape[0], cell_value.shape[1])
        for i in range(cell_value.shape[0]):
            for j in range(cell_value.shape[1]):
                if 0 < cell_value[i][j] <= 12:
                    plt.scatter(i, j, color='red', alpha=cell_value[i][j] / 12, marker='s')
                elif cell_value[i][j] > 12:
                    plt.scatter(i, j, color='red', alpha=1, marker='s')
        plt.title('Density Iter :{}'.format(self.timer))

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

    def plot_to_excel(self):
        Plot_Latitude_List = []
        Plot_Longitude_List = []
        Plot_Status_List = []
        cells = self.cells
        for i in range(1, cells.shape[0] - 1):
            for j in range(1, cells.shape[1] - 1):
                if cells[i, j] == 1:
                    Plot_Latitude_List.append(i * Plot_scale / Latitude_scale + Latitude_min)
                    Plot_Longitude_List.append(- j * Plot_scale / Longitude_scale + Longitude_min)
                    Plot_Status_List.append(cells[i, j])
        Data_Dict = {
            "Latitude": Plot_Latitude_List,
            "Longitude": Plot_Longitude_List,
            "Status": Plot_Status_List
        }
        Data_to_Save = DataFrame(Data_Dict)
        # print(Data_to_Save)
        Data_to_Save.to_excel('program_output_cell.xlsx')

    def density_plot_to_excel(self):
        Plot_Density_List = []
        cells = self.cells
        cell_value = self.cell_value
        for i in range(1, cells.shape[0] - 1):
            for j in range(1, cells.shape[1] - 1):
                Latitude_Converted = i * Plot_scale / Latitude_scale + Latitude_min
                Longitude_Converted = - j * Plot_scale / Longitude_scale + Longitude_min
                if cell_value[i][j] > 0:  # and self.query_baidu_api(Latitude_Converted, Longitude_Converted) != 0:
                    self.Density_Plot_Latitude_List.append(Latitude_Converted)
                    self.Density_Plot_Longitude_List.append(Longitude_Converted)
                    Plot_Density_List.append(cell_value[i][j])
        Data_Dict = {
            "Latitude": self.Density_Plot_Latitude_List,
            "Longitude": self.Density_Plot_Longitude_List,
            "Density": Plot_Density_List
        }
        self.Density_Plot_Latitude_List = []
        self.Density_Plot_Longitude_List = []
        Data_to_Save = DataFrame(Data_Dict)
        # print(Data_to_Save)
        Data_to_Save.to_excel('program_output_density.xlsx')

    def density_prediction(self):
        cells = self.cells
        for i in range(1, cells.shape[0] - 1):
            for j in range(1, cells.shape[1] - 1):
                if 1 <= i - 2 < cells.shape[0] - 1 and 1 <= j - 2 < cells.shape[1] - 1:
                    # 一个元胞的数值由其在该时刻本身的生死状态和周围的24个邻居的状态
                    neighbor = cells[i - 2:i + 3, j - 2:j + 3].reshape((-1,))
                    # 线性卷积函数，self.mask*neighbor
                    # 返回的数组长度为max(M,N)-min(M,N)+1,此时返回的是完全重叠的点。边缘的点无效。
                    neighbor_num = np.convolve(self.mask_density, neighbor, 'valid')[0]
                    # print(neighbor_num)
                    self.cell_value[i][j] = neighbor_num
        # print(self.cell_value)

    '''
    def query_baidu_api(self, Latitude_Converted, Longitude_Converted):
        ''''''
        调用百度地图API查询对应经纬度的地理信息
        如果返回为空值，说明位置在海上，则丢弃该数据
        ''''''

        # 计算校验SN（百度API文档说明需要此步骤）
        query = "/reverse_geocoding/v3/?ak={0}&output=json&coordtype=wgs84ll&location={1}, {2}".format(
            ak, Latitude_Converted, Longitude_Converted)
        encodedStr = urllib.parse.quote(query, safe="/:=&?#+!$,;'@()*[]")
        sn = hashlib.md5(urllib.parse.quote_plus(encodedStr + sk).encode()).hexdigest()
        # 使用requests获取返回的json
        response = requests.get("{0}{1}&sn={2}".format(url, query, sn))
        data = response.text
        # print(data)
        city_code = eval(data)['result']['cityCode']
        # print(city_code)
        return city_code
    '''

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
            self.density_prediction()
            self.plot_to_excel()
            self.density_plot_to_excel()
            # plt.pause(0.2)
            # 循环20次之后，停止程序
            if _ == 20:
                break
        plt.ioff()


if __name__ == '__main__':
    game = GameOfLife(cells_shape=(Plot_x_range + 2, Plot_y_range + 2), Point_count=Point_count)
    game.update_and_plot(200)
