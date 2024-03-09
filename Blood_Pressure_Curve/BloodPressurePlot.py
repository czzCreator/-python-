import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

fileName = '脓毒症lowess1.xls'

# 加载数据
df = pd.read_excel(fileName)
# 计算每个血压对应的病人生存率
df['mbp_groups'] = pd.cut(df['mbp'], range(0, 200, 2))
survival_rates = df.groupby('mbp_groups')['Mortality Rate (%)'].mean()

# 过滤掉NaN值
survival_rates = survival_rates.dropna()
print("survival_rates  not  include  NaN: ", survival_rates)

# 获取每个范围的最末值
mbp_range_endpoints = [interval.right for interval in survival_rates.index]

# 获取数据的最大值和最小值
max_value = df['mbp'].max()
min_value = df['mbp'].min()

print("max value: ", max_value)
print("min_value: ", min_value)

# 创建柱状图
plt.figure(figsize=(20, 10))
plt.bar(range(len(survival_rates)), survival_rates, color ='b', align ='center')

# 创建x轴标签 下面是两种标签样式 一种是直接显示区间的最大值 一种是显示大于最大值和小于最小值
##plt.xticks(range(len(survival_rates)), [str(interval) for interval in survival_rates.index], rotation =90)

# 创建x轴标签
all_x_labels = [str(endpoint) for endpoint in mbp_range_endpoints]
print("all_x_labels: ", all_x_labels)
# 将 柱状图中 y轴有值的非空的柱状的横坐标 做下判定，把这些横坐标的搜集到列表中
# 这个列表的作用是为了在x轴上显示的时候，只显示有值的柱状图的横坐标
bValidX_labels = [str(endpoint) for endpoint, survival_rate in zip(mbp_range_endpoints, survival_rates) if not np.isnan(survival_rate)]
print("bValidX_labels: ", bValidX_labels)
# 修改 bValidX_labels 的最大值和最小值
bValidX_labels[0] = '< {}'.format(bValidX_labels[1])
bValidX_labels[-1] = '> {}'.format(bValidX_labels[-2])
plt.xticks(range(len(survival_rates)), bValidX_labels)  #rotation=90

# 将纵坐标标签格式化为百分数
#plt.yticks(ticks=np.arange(0, 1.1, 0.1), labels=['{}%'.format(int(x*100)) for x in np.arange(0, 1.1, 0.1)])
# 计算刻度位置和标签
yticks_values = np.arange(0, 1.05, 0.05)
yticks_labels = ['{}%'.format(int(x*100)) for x in yticks_values]
# 设置刻度位置和标签
plt.yticks(ticks=yticks_values, labels=yticks_labels)

# 创建横坐标标题
plt.xlabel('Blood Pressure (mmHg)', fontdict={'fontsize': 14, 'fontweight': 'bold'})
# 创建纵坐标标题
plt.ylabel('Mortality Rate (%)', fontdict={'fontsize': 14, 'fontweight': 'bold'})
# 创建标题
##plt.title('Survival Rate by Blood Pressure', fontdict={'fontsize': 16, 'fontweight': 'bold'})

# 创建LOWESS平滑曲线，去掉红点
sns.regplot(x=np.array(range(len(survival_rates))), y=survival_rates, lowess=True, color='r',scatter=False)
# 显示图
plt.tight_layout()

dpiValue = 300
plt.savefig( fileName+'.eps', format='eps',dpi=dpiValue)
plt.savefig( fileName+'.png', dpi=dpiValue)    
plt.show()