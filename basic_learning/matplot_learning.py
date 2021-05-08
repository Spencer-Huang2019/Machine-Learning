import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


f = plt.figure()
ax = plt.axes(projection='3d')

xx = np.random.random(20)*10-5  #取100个随机数，范围在-5~5之间
yy = np.random.random(20)*10-5
X, Y = np.meshgrid(xx, yy)
Z = np.sin(np.sqrt(X**2+Y**2))

ax.scatter(X, Y, Z, alpha=0.3, c=np.random.random(400), s=np.random.randint(10, 20, size=(20, 20)))

plt.show()

# f = plt.figure()
# ax = plt.axes(projection='3d')
#
# xx = np.arange(-5, 5, 0.5)
# yy = np.arange(-5, 5, 0.5)
# X, Y = np.meshgrid(xx, yy)  #生产网络点坐标矩阵
# Z = np.sin(np.sqrt(X**2+Y**2))
#
# ax.plot_surface(X, Y, Z, alpha=0.3, cmap='winter')  #alpha 用于控制透明度
# ax.contour(X, Y, Z, zdir='z', offset=-3, cmap="rainbow")
# ax.contour(X, Y, Z, zdir='x', offset=-6, cmap="rainbow")
# ax.contour(X, Y, Z, zdir='y', offset=6, cmap="rainbow")
#
# ax.set_xlabel('X')
# ax.set_xlim(-6, 4)  #拉开坐标轴范围显示投影
# ax.set_ylabel('Y')
# ax.set_ylim(-4, 6)
# ax.set_zlabel('Z')
# ax.set_zlim(-3, 3)
# plt.show()


# f = plt.figure()
# ax1 = plt.axes(projection='3d')
#
# xx = np.arange(-5, 5, 0.5)
# yy = np.arange(-5, 5, 0.5)
# X, Y = np.meshgrid(xx, yy)  #生产网络点坐标矩阵
# Z = np.sin(X)+np.cos(Y)
#
# ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
# ax1.contour(X, Y, Z, zdir='z', offset=-2, cmap='rainbow') #等高线图，设置offset，为Z的最小值
# plt.show()

# fig = plt.figure()
# ax1 = Axes3D(fig)
#
# z = np.linspace(0, 13, 1000)
# x = 5*np.sin(z)
# y = 5*np.cos(z)
# zd = 13*np.random.random(100)
# xd = 5*np.sin(zd)
# yd = 5*np.cos(zd)
# ax1.scatter3D(xd, yd, zd, cmap='Blues')  #绘制散点图
# ax1.plot3D(x, y, z, 'gray')  #绘制空间曲线
#
# plt.show()


# data = [100, 500, 300]
# fig = plt.figure(dpi=80)
# plt.pie(data,
#         explode=[0.0,0.1,0.1],  #每个饼块离中心的距离
#         colors=['y', 'r', 'g'],  # 每个饼块的颜色
#         labels=['A part', 'B part', 'C part'],  #每个饼块的标签
#         labeldistance=1.2,   #每个饼块标签到中心的距离
#         autopct='%1.1f%%',  #百分比的显示格式
#         pctdistance=0.4,  #百分比到中心的距离
#         shadow=True,  #每个饼块是否显示阴影
#         startangle=0,  #默认从x轴正半轴逆时针起
#         radius=1  #饼块的半径
#         )
# plt.show()

# x = np.arange(1, 10)
# y = x
# fig = plt.figure()
# plt.scatter(x, y, c = 'r', marker = 'o') # marker指定三点形状为圆形
# plt.show()

# plt.figure(1, dpi=80)
# data = [1,1,1,2,2,2,3,3,4,6,4]
# plt.hist(data) # 直方图
#
# plt.show()

# x = np.linspace(-np.pi*2, np.pi*2, 100)
# plt.figure(1, dpi=80)
#
# plt.plot(x, np.sin(x), label='sin(x)')
# plt.plot(x, np.cos(x), label='cos(x)')
# plt.xlabel("X axe")
# plt.ylabel("Y axe")
# plt.title("sin(x) and cos(x) function")
# # plt.legend() #显示图例,这里会显示 哪条是sin(x), 哪条是cos(x)
# plt.show()


# plt.figure(1, dpi=80)
# ax1 = plt.subplot(211)
# ax2 = plt.subplot(212)
#
# x = np.linspace(0, 10, 100)
#
# plt.sca(ax1) # 选择子图 ax1
# plt.plot(x, np.exp(x))
#
# plt.sca(ax2)
# plt.plot(x, np.sin(x))
#
# plt.show()

# plt.figure(1, dpi=80)
# x = np.linspace(-np.pi, np.pi, 100) # x轴的定义域为 -3.14~3.14，中间间隔100个元素
# plt.plot(x, np.sin(x))
# plt.show()

# plt.figure(1, dpi=50) #dpi denote the size of figure
# plt.subplot(111)
# plt.figure(2, dpi=50)
# plt.subplot(223) #创建 2*2 的图表矩阵，绘制的子图为矩阵中的序号
# plt.show()