# import pandas as pd
#
# import matplotlib.pyplot as plt
#
# df = pd.read_excel('data/position_error.xls')
# df.head()
#
# x = df['error']
# x, y1, y2, y3,y4,y5 = df['error'], df['V2X-Mamba'], \
#     df['early_fusion'], df['fcooper'], df['opv2v'], df['v2xvit']
# # 准备数据
#
# # plt.figure(figsize=(5, 4))
#
# plt.plot(x, y1*100, label='v2x-Mamba', c='r',marker='*')
# plt.plot(x, y2*100, label='early_fusion', c='b',marker='*')
# plt.plot(x, y3*100, label='fcooper', c='g',marker='*')
# plt.plot(x, y4*100, label='opv2v', c='c',marker='*')
# plt.plot(x, y5*100, label='v2xvit', c='m',marker='*')
#
#
# plt.ylabel('Average Precision @ IoU=0.7 ')
# plt.xlabel('Position error std(m)')
#
#
# plt.legend(loc='lower left', fontsize=10)  # 图例
# plt.grid(axis='y')
#
# plt.savefig('data/position_error.png')
#
# plt.show()

import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_excel('data/heading_error.xls')
df.head()

x = df['error']
x, y1, y2, y3,y4,y5 = df['error'], df['V2X-Mamba'], \
    df['early_fusion'], df['fcooper'], df['opv2v'], df['v2xvit']
# 准备数据

# plt.figure(figsize=(7, 4))

plt.plot(x, y1*100, label='v2x-Mamba', c='r',marker='*')
plt.plot(x, y2*100, label='early_fusion', c='b',marker='*')
plt.plot(x, y3*100, label='fcooper', c='g',marker='*')
plt.plot(x, y4*100, label='opv2v', c='c',marker='*')
plt.plot(x, y5*100, label='v2xvit', c='m',marker='*')

# plt.ylim(40,60)
plt.ylabel('Average Precision @ IoU=0.7 ')
plt.xlabel('Heading error std(m)')


plt.legend(loc='lower left', fontsize=10)  # 图例
plt.grid(axis='y')

plt.savefig('data/heading_error.png')

plt.show()
