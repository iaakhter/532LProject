import re
import matplotlib.pyplot as plt
import numpy as np

D_A = []
G_A = []
D_B = []
G_B = []
epoch = []
with open("loss_log.txt") as f:
	for line in f:
		result = re.search('epoch: (.*) time:', line)
		epoch.append(re.split('\W+', result.group(1))[0])
		# print(re.split('\W+', result.group(1)))
		result = re.search('D_A: (.*) Cyc_B:', line)
		arr = result.group(1).split()
		D_A.append(float(arr[0]))
		G_A.append(float(arr[2]))
		D_B.append(float(arr[6]))
		G_B.append(float(arr[8]))


# plt.style.use('seaborn-whitegrid')
fig = plt.figure()
x = np.linspace(0, 1, len(D_A))
plt.plot(x, D_A, linewidth=0.5, label='rock discriminator')
plt.plot(x, G_A, linewidth=0.5, label='rock generator')
plt.plot(x, D_B, linewidth=0.5, label='classical music discriminator')
plt.plot(x, G_B, linewidth=0.5, label='classical music generator')	
# plt.xticks(x, epoch)
ax = plt.gca()
ax.locator_params(tight='true',nbins=13)
ax.tick_params(labelbottom='off')   

ax.set_xlim([0,1])
ax.set_ylim(ymin=0)
# ax.xaxis.set_major_locator(plt.MaxNLocator(30))
plt.legend(loc='upper left')

plt.show()





